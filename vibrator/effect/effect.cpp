/*
 * Copyright (c) 2020, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of The Linux Foundation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Changes from Qualcomm Innovation Center are provided under the following license:
 * Copyright (c) 2022-2023 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

/*
 * Awinic HapticHV backend for the QTI vibrator HAL.
 *
 * Builds on the stock effect.h (struct effect_stream and get_effect_stream()
 * are declared there, unchanged). Vibrator.cpp keeps including "effect.h";
 * this file implements how the Awinic backend fills the stream so the HAL
 * and the stock "aw-haptic-hv" kernel driver both work.
 *
 * Vibrator.cpp (USE_EFFECT_STREAM) uses the stream like this:
 *   - stream != NULL           -> effect/primitive is advertised as supported
 *   - the stream pointer       -> passed to the kernel as FF_PERIODIC custom_data
 *   - length * 1000 / rate + 1 -> play length in ms reported to the framework
 * haptic_hv.c (input_upload_effect) copy_from_user()s only the first int16 of
 * custom_data as the RAM waveform id, valid 1 <= id < ram_num.
 *
 * So every stream returned by this library has:
 *   effect_id    = RAM waveform id (first int16, little endian)
 *   play_rate_hz = kPlayRateHz
 *   length       = duration_ms - 1  (HAL then computes exactly duration_ms)
 *   data         = nullptr
 */

#include "effect.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <android/log.h>
#include <tinyxml2.h>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "awvibeffect", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "awvibeffect", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "awvibeffect", __VA_ARGS__)

/* The driver reads the first int16 of the stream as the waveform id. */
static_assert(offsetof(effect_stream, effect_id) == 0, "effect_id must be first");
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "effect_id is read as an int16 at offset 0; this relies on little endian"
#endif

namespace {

constexpr uint32_t kPrimitiveMask = 0x8000;   /* PRIMITIVE_ID_MASK in Vibrator.cpp */
constexpr int kEffectCount = 22;              /* Effect::CLICK .. Effect::TEXTURE_TICK */
constexpr int kPrimitiveCount = 9;            /* CompositePrimitive::NOOP .. LOW_TICK */
constexpr int kMaxWaves = 32;
constexpr size_t kRamShift = 4;               /* checksum(2) + base addr(2) */
constexpr int kUnset = -2;                    /* conf: not mentioned, use default */
constexpr int kNone = -1;                     /* conf: explicitly unsupported */

constexpr uint32_t kPlayRateHz = 1000;        /* HAL: length*1000/rate+1 → length+1 */
constexpr uint32_t kDefaultRamRateHz = 24000; /* driver does set_pwm(AW_PWM_24K) */
constexpr uint32_t kDefaultMarginMs = 2;
constexpr uint32_t kFallbackWaveMs = 20;      /* only when no bin could be read */

constexpr std::string_view kConfName = "awinic_haptic_primitives.xml";
constexpr std::string_view kRamName = "haptic_ram.bin"; /* same name the driver requests */
/* ODM overrides vendor when both are present. */
constexpr std::array<std::string_view, 2> kConfDirs = {
    "/odm/etc/",
    "/vendor/etc/",
};
constexpr std::array<std::string_view, 2> kFirmwareDirs = {
    "/odm/firmware/",
    "/vendor/firmware/",
};
constexpr std::array<std::string_view, 2> kRamNumPaths = {
    "/sys/class/leds/aw_vibrator/ram_num",
    "/sys/class/leds/vibrator/ram_num",
};

/* android.hardware.vibrator.Effect */
enum {
    EFF_CLICK = 0, EFF_TICK = 2, EFF_THUD = 3, EFF_POP = 4,
    EFF_HEAVY_CLICK = 5, EFF_TEXTURE_TICK = 21,
};
/* android.hardware.vibrator.CompositePrimitive */
enum {
    PRIM_CLICK = 1, PRIM_THUD = 2, PRIM_LIGHT_TICK = 7, PRIM_LOW_TICK = 8,
};

struct Config {
    uint32_t ram_rate_hz = kDefaultRamRateHz;
    uint32_t margin_ms = kDefaultMarginMs;
    int ram_num = 0;                       /* 0 = autodetect */
    std::string bin_path;
    std::array<int, kMaxWaves> wave_ms{};          /* 0 = derive from bin */
    std::array<int, kEffectCount> effect_map{};
    std::array<int, kPrimitiveCount> prim_map{};

    Config()
    {
        wave_ms.fill(0);
        effect_map.fill(kUnset);
        prim_map.fill(kUnset);
    }
};

struct RamInfo {
    int ram_num = 0;                       /* includes the reserved loop slot */
    std::array<int, kMaxWaves> bytes{};    /* waveform size in RAM bytes, 0 = unknown */
};

struct Table {
    std::array<effect_stream, kEffectCount> effects{};
    std::array<effect_stream, kPrimitiveCount> prims{};
    std::array<bool, kEffectCount> effect_ok{};
    std::array<bool, kPrimitiveCount> prim_ok{};
};

struct State {
    std::mutex lock;
    bool ready = false;
    bool attempted = false;
    bool fail_logged = false;
    std::chrono::steady_clock::time_point last_try{};
    Table table;
};

State &state()
{
    static State s;
    return s;
}

/* ---- helpers ------------------------------------------------------------ */

std::string conf_path()
{
    for (std::string_view dir : kConfDirs) {
        std::string path(dir);
        path += kConfName;
        if (std::ifstream(path).good())
            return path;
    }
    /* Neither exists yet; prefer odm so a later drop-in is found first. */
    std::string fallback(kConfDirs.front());
    fallback += kConfName;
    return fallback;
}

bool read_file(std::string_view path, std::vector<uint8_t> *out)
{
    std::ifstream in{std::string(path), std::ios::binary};
    if (!in)
        return false;

    out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (out->size() > (1u << 20)) {        /* RAM bins are ~1 KB; refuse junk */
        out->clear();
        return false;
    }
    return !out->empty();
}

constexpr uint32_t be16(const uint8_t *p)
{
    return (static_cast<uint32_t>(p[0]) << 8) | p[1];
}

/*
 * Same layout the driver validates in ram_load() and later uses in
 * get_ram_num() / container_update(). Only the first int16 of custom_data
 * is ever read by input_upload_effect() as the waveform id.
 */
bool parse_bin(const std::vector<uint8_t> &d, RamInfo *ram)
{
    *ram = {};
    if (d.size() < kRamShift + 7)
        return false;

    uint16_t sum = 0;
    for (size_t i = 2; i < d.size(); i++)
        sum = static_cast<uint16_t>(sum + d[i]);
    if (sum != static_cast<uint16_t>(be16(&d[0]))) {
        LOGW("RAM bin checksum mismatch (0x%04x vs 0x%04x)", sum, be16(&d[0]));
        return false;
    }

    const uint32_t base = be16(&d[2]);
    const uint8_t *r = &d[kRamShift];            /* RAM image after checksum+base */
    const size_t rlen = d.size() - kRamShift;
    const uint32_t first = be16(&r[1]);
    if (first <= base + 1)
        return false;

    /* Driver: ram_num = (first_wave_addr - base_addr - 1) / 4 */
    const int n = static_cast<int>((first - base - 1) / 4);
    if (n < 2 || n >= kMaxWaves || static_cast<size_t>(1 + 4 * n) > rlen)
        return false;

    ram->ram_num = n;
    for (int w = 1; w <= n; w++) {
        const uint8_t *e = &r[1 + 4 * (w - 1)];
        uint32_t start = be16(&e[0]), end = be16(&e[2]);
        ram->bytes[w] = (end >= start) ? static_cast<int>(end - start + 1) : 0;
    }
    return true;
}

/* Non-throwing integer parse for conf / sysfs tokens. */
int parse_int(std::string_view s, int fallback = 0)
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.remove_prefix(1);
    int v = fallback;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    (void)ptr;
    return (ec == std::errc{}) ? v : fallback;
}

int read_sysfs_ram_num()
{
    /*
     * Reading ram_num makes the driver stop playback (get_ram_num() does
     * play_stop), so this is only ever done once, from init, as a fallback.
     */
    for (std::string_view path : kRamNumPaths) {
        std::ifstream in;
        in.open(std::string{path});
        if (!in.is_open())
            continue;

        std::string line;
        if (!std::getline(in, line))
            continue;

        /* Sysfs may print "ram_num = N" or just "N". */
        size_t eq = line.find('=');
        std::string_view token = (eq != std::string::npos)
            ? std::string_view(line).substr(eq + 1)
            : std::string_view(line);
        int v = parse_int(token);
        if (v > 0)
            return v;
    }
    return 0;
}

int wave_attr(const tinyxml2::XMLElement *e)
{
    const char *w = e->Attribute("wave");
    if (!w)
        return kUnset;
    if (std::string_view(w) == "none")
        return kNone;
    return parse_int(w, kNone);
}

void load_conf(Config *c)
{
    const std::string path = conf_path();
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
        /* Optional file — silent if missing, warn if present but broken. */
        if (std::ifstream(path).good())
            LOGW("failed to parse %s", path.c_str());
        return;
    }

    const tinyxml2::XMLElement *root = doc.RootElement();
    if (!root || std::string_view(root->Name()) != "haptic") {
        LOGW("%s: expected root <haptic>", path.c_str());
        return;
    }

    if (const tinyxml2::XMLElement *ram = root->FirstChildElement("ram")) {
        unsigned v = 0;
        if (ram->QueryUnsignedAttribute("rate_hz", &v) == tinyxml2::XML_SUCCESS &&
            v >= 1000 && v <= 200000)
            c->ram_rate_hz = v;
        if (ram->QueryUnsignedAttribute("margin_ms", &v) == tinyxml2::XML_SUCCESS && v <= 100)
            c->margin_ms = v;
        if (ram->QueryUnsignedAttribute("num", &v) == tinyxml2::XML_SUCCESS &&
            v >= 2 && v < kMaxWaves)
            c->ram_num = static_cast<int>(v);
        if (const char *bin = ram->Attribute("bin"))
            c->bin_path = bin;
    }

    for (const tinyxml2::XMLElement *e = root->FirstChildElement("wave"); e;
         e = e->NextSiblingElement("wave")) {
        unsigned id = 0, ms = 0;
        if (e->QueryUnsignedAttribute("id", &id) != tinyxml2::XML_SUCCESS ||
            e->QueryUnsignedAttribute("ms", &ms) != tinyxml2::XML_SUCCESS)
            continue;
        if (id >= 1 && id < kMaxWaves && ms >= 1 && ms <= 5000)
            c->wave_ms[id] = static_cast<int>(ms);
    }

    for (const tinyxml2::XMLElement *e = root->FirstChildElement("effect"); e;
         e = e->NextSiblingElement("effect")) {
        unsigned id = 0;
        if (e->QueryUnsignedAttribute("id", &id) != tinyxml2::XML_SUCCESS || id >= kEffectCount)
            continue;
        int w = wave_attr(e);
        if (w != kUnset)
            c->effect_map[id] = w;
    }

    for (const tinyxml2::XMLElement *e = root->FirstChildElement("primitive"); e;
         e = e->NextSiblingElement("primitive")) {
        unsigned id = 0;
        if (e->QueryUnsignedAttribute("id", &id) != tinyxml2::XML_SUCCESS ||
            id >= kPrimitiveCount)
            continue;
        int w = wave_attr(e);
        if (w != kUnset)
            c->prim_map[id] = w;
    }
}

uint32_t wave_ms(const Config &c, const RamInfo &r, int w)
{
    if (c.wave_ms[w] > 0)
        return static_cast<uint32_t>(c.wave_ms[w]);        /* user value is final */
    uint32_t ms = kFallbackWaveMs;
    if (r.bytes[w] > 0)
        ms = static_cast<uint32_t>(((static_cast<uint64_t>(r.bytes[w]) * 1000 +
                                     c.ram_rate_hz - 1) / c.ram_rate_hz));
    return std::max(ms + c.margin_ms, 1u);
}

/*
 * HAL computes: playLengthMs = length * 1000 / play_rate_hz + 1.
 * With rate = 1000 that is length + 1, so length = ms - 1 gives exactly ms.
 */
void fill_stream(effect_stream *s, int wave, uint32_t ms)
{
    s->effect_id = static_cast<uint32_t>(wave);  /* low int16 == driver waveform id */
    s->play_rate_hz = kPlayRateHz;
    s->length = ms - 1;
    s->data = nullptr;
}

bool wave_ok(const RamInfo &r, int w)
{
    return w >= 1 && w < r.ram_num;     /* ram_num itself is the reserved loop slot */
}

/*
 * Pick light / mid / heavy one-shots by duration without allocating.
 * Driver accepts only 1 <= id < ram_num (ram_num is the reserved loop slot).
 */
void pick_default_waves(const Config &c, const RamInfo &r,
                        int *light, int *mid, int *heavy)
{
    std::array<int, kMaxWaves> shots{};
    int n = 0;
    for (int w = 1; w < r.ram_num; w++)
        shots[n++] = w;

    std::sort(shots.begin(), shots.begin() + n, [&](int a, int b) {
        return wave_ms(c, r, a) < wave_ms(c, r, b);
    });
    *light = shots[0];
    *mid = shots[n / 2];
    *heavy = shots[n - 1];
}

void map_one(const Config &c, const RamInfo &r, const char *kind, int id,
             int wave, effect_stream *stream, bool *ok)
{
    if (wave != kNone && !wave_ok(r, wave)) {
        LOGW("%s %d -> wave %d is not a valid RAM one-shot (1..%d), disabled",
             kind, id, wave, r.ram_num - 1);
        wave = kNone;
    }
    if (wave != kNone) {
        fill_stream(stream, wave, wave_ms(c, r, wave));
        *ok = true;
    }
}

void build_table(const Config &c, const RamInfo &r, Table *t)
{
    *t = {};

    int light, mid, heavy;
    pick_default_waves(c, r, &light, &mid, &heavy);

    /*
     * One waveform per effect/primitive. Anything that needs several pulses
     * or a ramp (DOUBLE_CLICK, SPIN, *_RISE, QUICK_FALL, RINGTONE_*) or
     * silence (NOOP) stays unsupported unless the conf explicitly maps it.
     */
    std::array<int, kEffectCount> effect_def{};
    std::array<int, kPrimitiveCount> prim_def{};
    effect_def.fill(kNone);
    prim_def.fill(kNone);

    effect_def[EFF_CLICK] = mid;
    effect_def[EFF_TICK] = light;
    effect_def[EFF_THUD] = heavy;
    effect_def[EFF_POP] = light;
    effect_def[EFF_HEAVY_CLICK] = heavy;
    effect_def[EFF_TEXTURE_TICK] = light;

    prim_def[PRIM_CLICK] = mid;
    prim_def[PRIM_THUD] = heavy;
    prim_def[PRIM_LIGHT_TICK] = light;
    prim_def[PRIM_LOW_TICK] = mid;

    for (int i = 0; i < kEffectCount; i++) {
        int mapped = (c.effect_map[i] == kUnset) ? effect_def[i] : c.effect_map[i];
        map_one(c, r, "effect", i, mapped, &t->effects[i], &t->effect_ok[i]);
    }
    for (int i = 0; i < kPrimitiveCount; i++) {
        int mapped = (c.prim_map[i] == kUnset) ? prim_def[i] : c.prim_map[i];
        map_one(c, r, "primitive", i, mapped, &t->prims[i], &t->prim_ok[i]);
    }
}

bool init(State *s)
{
    Config c;
    RamInfo r;
    std::vector<uint8_t> bin;
    bool have_bin = false;

    load_conf(&c);

    if (!c.bin_path.empty())
        have_bin = read_file(c.bin_path, &bin) && parse_bin(bin, &r);
    for (std::string_view dir : kFirmwareDirs) {
        if (have_bin)
            break;
        std::string path(dir);
        path += kRamName;
        have_bin = read_file(path, &bin) && parse_bin(bin, &r);
    }

    if (c.ram_num) {
        r.ram_num = c.ram_num;                       /* explicit override wins */
    } else if (!have_bin) {
        r.ram_num = read_sysfs_ram_num();
    }
    if (r.ram_num < 2) {
        if (!s->fail_logged)
            LOGE("no RAM waveform info (no readable haptic_ram.bin, ram_num unavailable)");
        s->fail_logged = true;
        return false;
    }

    build_table(c, r, &s->table);
    LOGI("ram_num=%d (%s), rate=%uHz", r.ram_num,
         c.ram_num ? "conf" : have_bin ? "bin" : "sysfs", c.ram_rate_hz);
    return true;
}

} // namespace

const struct effect_stream *get_effect_stream(uint32_t effect_id)
{
    State &s = state();
    std::lock_guard<std::mutex> l(s.lock);

    if (!s.ready) {
        /* Don't hammer the filesystem if the RAM info isn't there yet. */
        auto now = std::chrono::steady_clock::now();
        if (s.attempted && now - s.last_try < std::chrono::seconds(1))
            return nullptr;
        s.attempted = true;
        s.last_try = now;
        if (!(s.ready = init(&s)))
            return nullptr;
    }

    if (effect_id & kPrimitiveMask) {
        uint32_t p = effect_id & ~kPrimitiveMask;
        if (p < static_cast<uint32_t>(kPrimitiveCount) && s.table.prim_ok[p])
            return &s.table.prims[p];
    } else if (effect_id < static_cast<uint32_t>(kEffectCount) && s.table.effect_ok[effect_id]) {
        return &s.table.effects[effect_id];
    }
    return nullptr;
}

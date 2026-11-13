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
 * Populates effect_stream entries for get_effect_stream() declared in effect.h.
 * Vibrator.cpp uses a non-null stream pointer as FF_PERIODIC custom_data, where
 * the driver reads only the first int16 (effect_id) as the RAM waveform ID,
 * and computes reported duration as length * 1000 / play_rate_hz + 1.
 */

#include "effect.h"

#include <android/log.h>
#include <tinyxml2.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#define LOGE(...) \
    do { \
        __android_log_print(ANDROID_LOG_ERROR, "awvibeffect", __VA_ARGS__); \
    } while (0)
#define LOGW(...) \
    do { \
        __android_log_print(ANDROID_LOG_WARN, "awvibeffect", __VA_ARGS__); \
    } while (0)
#define LOGI(...) \
    do { \
        __android_log_print(ANDROID_LOG_INFO, "awvibeffect", __VA_ARGS__); \
    } while (0)

/* The driver reads the first int16 of effect_stream as the RAM waveform ID. */
static_assert(offsetof(effect_stream, effect_id) == 0, "effect_id must be first");
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "effect_id is read as an int16 at offset 0; this relies on little endian"
#endif

namespace {

constexpr uint32_t kPrimitiveMask = 0x8000;
constexpr int kEffectCount = 22;
constexpr int kPrimitiveCount = 9;
constexpr int kMaxWaves = 32;
/* Header layout before RAM image: 2-byte checksum + 2-byte base address. */
constexpr size_t kRamShift = 4;
constexpr int kUnset = -2;
constexpr int kNone = -1;

/* With play_rate_hz = 1000, HAL playLengthMs = length + 1, so length = ms - 1. */
constexpr uint32_t kPlayRateHz = 1000;
constexpr uint32_t kDefaultRamRateHz = 24000;
constexpr uint32_t kDefaultMarginMs = 2;
constexpr uint32_t kFallbackWaveMs = 20;

constexpr std::string_view kConfName = "awinic_haptic_primitives.xml";
constexpr std::string_view kRamName = "haptic_ram.bin";
/* ODM overrides vendor when both exist. */
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

/* android.hardware.vibrator.Effect indices */
enum {
    EFF_CLICK = 0,
    EFF_TICK = 2,
    EFF_THUD = 3,
    EFF_POP = 4,
    EFF_HEAVY_CLICK = 5,
    EFF_TEXTURE_TICK = 21,
};

/* android.hardware.vibrator.CompositePrimitive indices */
enum {
    PRIM_CLICK = 1,
    PRIM_THUD = 2,
    PRIM_LIGHT_TICK = 7,
    PRIM_LOW_TICK = 8,
};

/* first: ram_num (includes reserved loop slot), second: byte length per wave ID. */
using RamBinData = std::pair<int, std::array<int, kMaxWaves>>;
using WaveArray = std::array<int, kMaxWaves>;
using EffectArray = std::array<int, kEffectCount>;
using PrimitiveArray = std::array<int, kPrimitiveCount>;

std::mutex g_lock;
int g_state = 1; /* >0: unattempted, 0: ready, <0: failed */
std::chrono::steady_clock::time_point g_last_try{};
std::array<effect_stream, kEffectCount> g_effects{};
std::array<effect_stream, kPrimitiveCount> g_prims{};
std::array<bool, kEffectCount> g_effect_ok{};
std::array<bool, kPrimitiveCount> g_prim_ok{};

constexpr uint32_t ParseBe16(const uint8_t* bytes) {
    return (static_cast<uint32_t>(bytes[0]) << 8) | bytes[1];
}

int ParseInt(std::string_view text, int fallback = 0) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' ||
                             text.front() == '\r' || text.front() == '\n')) {
        text.remove_prefix(1);
    }
    int value = fallback;
    auto [ptr, error_code] = std::from_chars(text.data(), text.data() + text.size(), value);
    (void)ptr;
    return (error_code == std::errc{}) ? value : fallback;
}

int ParseWaveAttr(const tinyxml2::XMLElement* elem) {
    const char* wave_str = elem->Attribute("wave");
    if (!wave_str) {
        return kUnset;
    }
    if (std::string_view(wave_str) == "none") {
        return kNone;
    }
    return ParseInt(wave_str, kNone);
}

/*
 * Validates and extracts waveform sizes from haptic_ram.bin using the layout
 * expected by the aw-haptic-hv driver (ram_load / container_update).
 */
bool ParseRamBin(std::string_view path, RamBinData* ram_info) {
    *ram_info = {};
    std::ifstream in{std::string(path), std::ios::binary};
    if (!in) {
        return false;
    }

    std::vector<uint8_t> bin_data{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>{}};
    if (bin_data.size() < kRamShift + 7 || bin_data.size() > (1u << 20)) {
        return false;
    }

    uint16_t sum = 0;
    for (size_t i = 2; i < bin_data.size(); ++i) {
        sum += bin_data[i];
    }
    if (sum != static_cast<uint16_t>(ParseBe16(&bin_data[0]))) {
        LOGW("RAM bin checksum mismatch (0x%04x vs 0x%04x)", sum, ParseBe16(&bin_data[0]));
        return false;
    }

    const uint32_t base = ParseBe16(&bin_data[2]);
    const uint8_t* ram_image = &bin_data[kRamShift];
    const size_t ram_len = bin_data.size() - kRamShift;
    const uint32_t first = ParseBe16(&ram_image[1]);
    if (first <= base + 1) {
        return false;
    }

    /* Driver formula: ram_num = (first_wave_addr - base_addr - 1) / 4. */
    const int wave_count = static_cast<int>((first - base - 1) / 4);
    if (wave_count < 2 || wave_count >= kMaxWaves ||
        static_cast<size_t>(1 + 4 * wave_count) > ram_len) {
        return false;
    }

    ram_info->first = wave_count;
    for (int w = 1; w <= wave_count; ++w) {
        const uint8_t* entry = &ram_image[1 + 4 * (w - 1)];
        const uint32_t start = ParseBe16(&entry[0]);
        const uint32_t end = ParseBe16(&entry[2]);
        ram_info->second[w] = (end >= start) ? static_cast<int>(end - start + 1) : 0;
    }
    return true;
}

uint32_t ComputeWaveMs(const WaveArray& wave_ms, const RamBinData& ram_info,
                       uint32_t ram_rate_hz, uint32_t margin_ms, int wave) {
    if (wave < 0 || wave >= kMaxWaves) {
        return kFallbackWaveMs;
    }
    if (wave_ms[wave] > 0) {
        return static_cast<uint32_t>(wave_ms[wave]);
    }
    uint32_t ms = kFallbackWaveMs;
    if (ram_info.second[wave] > 0) {
        /* Ceiling division of (bytes * 1000) / ram_rate_hz. */
        ms = static_cast<uint32_t>(
            (static_cast<uint64_t>(ram_info.second[wave]) * 1000 + ram_rate_hz - 1) /
            ram_rate_hz);
    }
    return std::max(ms + margin_ms, 1u);
}

void BindStream(const WaveArray& wave_ms, const RamBinData& ram_info, uint32_t ram_rate_hz,
                uint32_t margin_ms, const char* kind, int id, int wave, effect_stream* stream,
                bool* ok) {
    /* ram_info.first (ram_num) is the reserved loop slot; only 1..ram_num-1 are valid. */
    if (wave != kNone && (wave < 1 || wave >= ram_info.first)) {
        LOGW("%s %d -> wave %d is not a valid RAM one-shot (1..%d), disabled", kind, id, wave,
             ram_info.first - 1);
        wave = kNone;
    } else if (wave != kNone) {
        stream->effect_id = static_cast<uint32_t>(wave);
        stream->play_rate_hz = kPlayRateHz;
        stream->length = ComputeWaveMs(wave_ms, ram_info, ram_rate_hz, margin_ms, wave) - 1;
        stream->data = nullptr;
        *ok = true;
    }
}

std::string ResolveConfPath() {
    for (std::string_view dir : kConfDirs) {
        std::string candidate(dir);
        candidate += kConfName;
        if (std::ifstream(candidate).good()) {
            return candidate;
        }
    }
    std::string fallback(kConfDirs.front());
    fallback += kConfName;
    return fallback;
}

void ParseXmlConfig(uint32_t* ram_rate_hz, uint32_t* margin_ms, int* conf_ram_num,
                    std::string* bin_path, WaveArray* wave_ms, EffectArray* effect_map,
                    PrimitiveArray* prim_map, int* role_light, int* role_mid, int* role_heavy) {
    const std::string conf_path = ResolveConfPath();
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(conf_path.c_str()) != tinyxml2::XML_SUCCESS) {
        if (doc.ErrorID() != tinyxml2::XML_ERROR_FILE_NOT_FOUND) {
            LOGW("failed to parse %s", conf_path.c_str());
        }
        return;
    }

    const tinyxml2::XMLElement* root = doc.RootElement();
    if (!root || std::string_view(root->Name()) != "haptic") {
        LOGW("%s: expected root <haptic>", conf_path.c_str());
        return;
    }

    if (const tinyxml2::XMLElement* ram = root->FirstChildElement("ram")) {
        unsigned value = 0;
        if (ram->QueryUnsignedAttribute("rate_hz", &value) == tinyxml2::XML_SUCCESS &&
            value >= 1000 && value <= 200000) {
            *ram_rate_hz = value;
        }
        if (ram->QueryUnsignedAttribute("margin_ms", &value) == tinyxml2::XML_SUCCESS &&
            value <= 100) {
            *margin_ms = value;
        }
        if (ram->QueryUnsignedAttribute("num", &value) == tinyxml2::XML_SUCCESS &&
            value >= 2 && value < static_cast<unsigned>(kMaxWaves)) {
            *conf_ram_num = static_cast<int>(value);
        }
        if (const char* bin = ram->Attribute("bin")) {
            *bin_path = bin;
        }
    }

    if (const tinyxml2::XMLElement* waves = root->FirstChildElement("waves")) {
        for (const tinyxml2::XMLElement* elem = waves->FirstChildElement("wave"); elem;
             elem = elem->NextSiblingElement("wave")) {
            unsigned id = 0;
            unsigned ms = 0;
            if (elem->QueryUnsignedAttribute("id", &id) != tinyxml2::XML_SUCCESS ||
                elem->QueryUnsignedAttribute("ms", &ms) != tinyxml2::XML_SUCCESS) {
                continue;
            }
            if (id >= 1 && id < static_cast<unsigned>(kMaxWaves) && ms >= 1 && ms <= 5000) {
                (*wave_ms)[id] = static_cast<int>(ms);
                if (const char* role = elem->Attribute("role")) {
                    std::string_view role_view(role);
                    if (role_view == "light") {
                        *role_light = static_cast<int>(id);
                    } else if (role_view == "mid") {
                        *role_mid = static_cast<int>(id);
                    } else if (role_view == "heavy") {
                        *role_heavy = static_cast<int>(id);
                    }
                }
            }
        }
    }

    if (const tinyxml2::XMLElement* effects = root->FirstChildElement("effects")) {
        for (const tinyxml2::XMLElement* elem = effects->FirstChildElement("effect"); elem;
             elem = elem->NextSiblingElement("effect")) {
            unsigned id = 0;
            if (elem->QueryUnsignedAttribute("id", &id) != tinyxml2::XML_SUCCESS ||
                id >= static_cast<unsigned>(kEffectCount)) {
                continue;
            }
            int wave = ParseWaveAttr(elem);
            if (wave != kUnset) {
                (*effect_map)[id] = wave;
            }
        }
    }

    if (const tinyxml2::XMLElement* primitives = root->FirstChildElement("primitives")) {
        for (const tinyxml2::XMLElement* elem = primitives->FirstChildElement("primitive"); elem;
             elem = elem->NextSiblingElement("primitive")) {
            unsigned id = 0;
            if (elem->QueryUnsignedAttribute("id", &id) != tinyxml2::XML_SUCCESS ||
                id >= static_cast<unsigned>(kPrimitiveCount)) {
                continue;
            }
            int wave = ParseWaveAttr(elem);
            if (wave != kUnset) {
                (*prim_map)[id] = wave;
            }
        }
    }
}

bool ResolveRamBin(std::string_view bin_path, RamBinData* ram_info) {
    if (!bin_path.empty() && ParseRamBin(bin_path, ram_info)) {
        return true;
    }
    for (std::string_view dir : kFirmwareDirs) {
        std::string path(dir);
        path += kRamName;
        if (ParseRamBin(path, ram_info)) {
            return true;
        }
    }
    return false;
}

int ReadSysfsRamNum() {
    /* Reading sysfs ram_num calls play_stop() in the driver, so do this once. */
    for (std::string_view path : kRamNumPaths) {
        std::ifstream in{std::string(path)};
        if (!in.is_open()) {
            continue;
        }

        std::string line;
        if (!std::getline(in, line)) {
            continue;
        }

        size_t eq_pos = line.find('=');
        std::string_view token = (eq_pos != std::string::npos)
                                     ? std::string_view(line).substr(eq_pos + 1)
                                     : line;
        int value = ParseInt(token);
        if (value > 0) {
            return value;
        }
    }
    return 0;
}

void SelectDefaultWaves(const WaveArray& wave_ms, const RamBinData& ram_info,
                        uint32_t ram_rate_hz, uint32_t margin_ms, int role_light, int role_mid,
                        int role_heavy, int* light, int* mid, int* heavy) {
    /* Sort available one-shot waves by duration as fallback for omitted role attributes. */
    WaveArray shots{};
    int shot_count = 0;
    const int upper = std::min(ram_info.first, kMaxWaves);
    for (int w = 1; w < upper; ++w) {
        shots[shot_count++] = w;
    }

    std::sort(shots.begin(), shots.begin() + shot_count, [&](int left, int right) {
        return ComputeWaveMs(wave_ms, ram_info, ram_rate_hz, margin_ms, left) <
               ComputeWaveMs(wave_ms, ram_info, ram_rate_hz, margin_ms, right);
    });
    *light = (role_light > 0) ? role_light : shots[0];
    *mid = (role_mid > 0) ? role_mid : shots[shot_count / 2];
    *heavy = (role_heavy > 0) ? role_heavy : shots[shot_count - 1];
}

void InitEffectStreams() {
    uint32_t ram_rate_hz = kDefaultRamRateHz;
    uint32_t margin_ms = kDefaultMarginMs;
    int conf_ram_num = 0;
    int role_light = 0;
    int role_mid = 0;
    int role_heavy = 0;
    std::string bin_path;
    WaveArray wave_ms{};
    EffectArray effect_map{
        kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset,
        kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset,
    };
    PrimitiveArray prim_map{
        kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset, kUnset,
    };
    RamBinData ram_info{};

    ParseXmlConfig(&ram_rate_hz, &margin_ms, &conf_ram_num, &bin_path, &wave_ms, &effect_map,
                   &prim_map, &role_light, &role_mid, &role_heavy);

    const bool have_bin = ResolveRamBin(bin_path, &ram_info);
    if (conf_ram_num) {
        ram_info.first = conf_ram_num;
    } else if (!have_bin) {
        ram_info.first = ReadSysfsRamNum();
    }

    if (ram_info.first < 2) {
        if (g_state >= 0) {
            LOGE("no RAM waveform info (no readable haptic_ram.bin, ram_num unavailable)");
        }
        g_state = -1;
        return;
    }

    g_effects.fill({});
    g_prims.fill({});
    g_effect_ok.fill(false);
    g_prim_ok.fill(false);

    int light = 0;
    int mid = 0;
    int heavy = 0;
    SelectDefaultWaves(wave_ms, ram_info, ram_rate_hz, margin_ms, role_light, role_mid,
                       role_heavy, &light, &mid, &heavy);

    /* Multi-pulse, ramp, and NOOP entries stay disabled unless explicitly mapped. */
    EffectArray effect_def{
        kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone,
        kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone,
    };
    PrimitiveArray prim_def{
        kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone, kNone,
    };

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

    for (int i = 0; i < kEffectCount; ++i) {
        int mapped = (effect_map[i] == kUnset) ? effect_def[i] : effect_map[i];
        BindStream(wave_ms, ram_info, ram_rate_hz, margin_ms, "effect", i, mapped, &g_effects[i],
                   &g_effect_ok[i]);
    }
    for (int i = 0; i < kPrimitiveCount; ++i) {
        int mapped = (prim_map[i] == kUnset) ? prim_def[i] : prim_map[i];
        BindStream(wave_ms, ram_info, ram_rate_hz, margin_ms, "primitive", i, mapped, &g_prims[i],
                   &g_prim_ok[i]);
    }

    g_state = 0;
    LOGI("ram_num=%d (%s), rate=%uHz", ram_info.first,
         conf_ram_num ? "conf" : (have_bin ? "bin" : "sysfs"), ram_rate_hz);
}

}  // namespace

const effect_stream* get_effect_stream(uint32_t effect_id) {
    std::lock_guard<std::mutex> guard(g_lock);

    if (g_state != 0) {
        /* Rate-limit filesystem probes if firmware/sysfs is not ready yet. */
        auto now = std::chrono::steady_clock::now();
        if (g_state < 0 && now - g_last_try < std::chrono::seconds(1)) {
            return nullptr;
        }
        g_last_try = now;
        InitEffectStreams();
        if (g_state != 0) {
            return nullptr;
        }
    }

    if (effect_id & kPrimitiveMask) {
        uint32_t prim_id = effect_id & ~kPrimitiveMask;
        if (prim_id < static_cast<uint32_t>(kPrimitiveCount) && g_prim_ok[prim_id]) {
            return &g_prims[prim_id];
        }
    } else if (effect_id < static_cast<uint32_t>(kEffectCount) && g_effect_ok[effect_id]) {
        return &g_effects[effect_id];
    }
    return nullptr;
}

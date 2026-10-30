#! /vendor/bin/sh
#
# SPDX-FileCopyrightText: The LineageOS Project
# SPDX-License-Identifier: Apache-2.0
#

soc_sku=`cat /sys/devices/soc0/sku`
function set_soc_sku() {
    case "$soc_sku" in
        "VOLCANO-1-AC")
            setprop ro.vendor.qti.soc_sku "SM7635Pro"
            ;;
        "VOLCANO-1-AB")
            setprop ro.vendor.qti.soc_sku "SM7635"
            ;;
        *)
            ;;
    esac
}
set_soc_sku

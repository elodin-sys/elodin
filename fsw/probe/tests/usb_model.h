// STM32 USB device register/PMA model shared by the USB host tests.
#ifndef USB_MODEL_H
#define USB_MODEL_H

#include "test.h"

#include <vector>

#define HAL_H
typedef U8 Hal_Irq;

static bool arch_irq_valid(Hal_Irq irq) {
    return irq <= 30u;
}

static void arch_irq_set_priority(Hal_Irq irq, U8 priority) {
    (void)irq;
    (void)priority;
}

static void arch_irq_enable(Hal_Irq irq) {
    (void)irq;
}

static void arch_irq_disable(Hal_Irq irq) {
    (void)irq;
}

static void arch_barrier(void) {
}

class ModelRegister {
public:
    ModelRegister(void *device, U32 offset) : device_(device), offset_(offset) {}
    operator U32() const;
    ModelRegister &operator=(U32 value);
    ModelRegister &operator|=(U32 value) {
        return *this = (U32)*this | value;
    }
    ModelRegister &operator&=(U32 value) {
        return *this = (U32)*this & value;
    }

private:
    void *device_;
    U32 offset_;
};

#define USBD__REG(dev_, offset_) ModelRegister((void *)(dev_), (U32)(offset_))
#include "../usbd.h"
#include "../dap_usb.h"
#include "../usb_dfu.h"

namespace {

constexpr U32 EndpointToggleMask = USBD__EP_DTOG_RX |
                                   USBD__EP_STAT_RX_MASK |
                                   USBD__EP_DTOG_TX |
                                   USBD__EP_STAT_TX_MASK;
constexpr U32 EndpointCtrMask = USBD__EP_CTR_RX | USBD__EP_CTR_TX;
constexpr U32 EndpointReadOnlyMask = USBD__EP_SETUP;
constexpr U32 IstrEventMask = USBD__ISTR_RESET | USBD__ISTR_SUSP |
                              USBD__ISTR_WKUP | USBD__ISTR_ERR;

struct UsbModel {
    alignas(4) U8 pma[2048]{};
    U32 regs[0x60 / 4]{};
    U32 endpoints[8]{};

    U32 read(U32 offset) const {
        if (offset < 8u * 4u) return endpoints[offset / 4u];
        if (offset != USBD__ISTR) return regs[offset / 4u];

        U32 result = regs[USBD__ISTR / 4u] & IstrEventMask;
        for (U8 ep = 0u; ep < 8u; ep++) {
            U32 value = endpoints[ep];
            if ((value & EndpointCtrMask) == 0u) continue;
            result |= USBD__ISTR_CTR | ep;
            if ((value & USBD__EP_CTR_RX) != 0u) result |= USBD__ISTR_DIR;
            break;
        }
        return result;
    }

    void write(U32 offset, U32 value) {
        if (offset < 8u * 4u) {
            U8 ep = (U8)(offset / 4u);
            U32 old = endpoints[ep];
            U32 regular = ~(EndpointToggleMask | EndpointCtrMask |
                            EndpointReadOnlyMask);
            U32 next = value & regular;
            next |= (old ^ value) & EndpointToggleMask;
            next |= old & value & EndpointCtrMask;
            if ((next & USBD__EP_CTR_RX) != 0u) {
                next |= old & USBD__EP_SETUP;
            }
            endpoints[ep] = next;
            return;
        }
        if (offset == USBD__ISTR) {
            regs[offset / 4u] &= value & IstrEventMask;
            return;
        }
        regs[offset / 4u] = value;
    }

    U32 &descriptor(U8 ep, bool rx) {
        return *reinterpret_cast<U32 *>(pma + ep * 8u + (rx ? 4u : 0u));
    }

    U16 buffer_address(U8 ep, bool rx) {
        return (U16)descriptor(ep, rx);
    }

    U16 buffer_count(U8 ep, bool rx) {
        return (U16)((descriptor(ep, rx) >> 16) & 0x03FFu);
    }

    U16 receive_capacity(U8 ep) {
        U16 value = (U16)(descriptor(ep, true) >> 16);
        U16 blocks = (U16)((value >> 10) & 0x1Fu);
        if ((value & 0x8000u) != 0u) return (U16)((blocks + 1u) * 32u);
        return (U16)(blocks * 2u);
    }

    void set_received_count(U8 ep, U16 count) {
        U32 &rx = descriptor(ep, true);
        rx = (rx & 0xFC00FFFFu) | ((U32)(count & 0x03FFu) << 16);
    }

    void set_status(U8 ep, U32 mask, U32 status) {
        endpoints[ep] = (endpoints[ep] & ~mask) | status;
    }

    void bus_reset(Usbd_Device *device) {
        regs[USBD__ISTR / 4u] |= USBD__ISTR_RESET;
        usbd_irq(device);
    }

    void suspend(Usbd_Device *device) {
        regs[USBD__ISTR / 4u] |= USBD__ISTR_SUSP;
        usbd_irq(device);
    }

    void wakeup(Usbd_Device *device) {
        regs[USBD__ISTR / 4u] |= USBD__ISTR_WKUP;
        usbd_irq(device);
    }

    // deliver = false: received, interrupt not yet serviced.
    void out(Usbd_Device *device, U8 ep, U8 const *data, U16 size,
             bool setup = false, bool deliver = true) {
        if (!setup) {
            CHECK((endpoints[ep] & USBD__EP_STAT_RX_MASK) ==
                  USBD__EP_STAT_RX_VALID);
        }
        CHECK(size <= receive_capacity(ep));
        if (size != 0u) {
            memcpy(pma + buffer_address(ep, true), data, size);
        }
        set_received_count(ep, size);
        set_status(ep, USBD__EP_STAT_RX_MASK, USBD__EP_STAT_RX_NAK);
        endpoints[ep] ^= USBD__EP_DTOG_RX;
        endpoints[ep] |= USBD__EP_CTR_RX;
        if (setup) endpoints[ep] |= USBD__EP_SETUP;
        if (deliver) usbd_irq(device);
    }

    std::vector<U8> in(Usbd_Device *device, U8 ep) {
        CHECK((endpoints[ep] & USBD__EP_STAT_TX_MASK) ==
              USBD__EP_STAT_TX_VALID);
        U16 size = buffer_count(ep, false);
        U16 address = buffer_address(ep, false);
        std::vector<U8> data(size);
        if (size != 0u) memcpy(data.data(), pma + address, size);
        set_status(ep, USBD__EP_STAT_TX_MASK, USBD__EP_STAT_TX_NAK);
        endpoints[ep] ^= USBD__EP_DTOG_TX;
        endpoints[ep] |= USBD__EP_CTR_TX;
        usbd_irq(device);
        return data;
    }
};

UsbModel *active_model;

std::vector<U8> setup_bytes(U8 request_type, U8 request, U16 value,
                            U16 index, U16 length) {
    std::vector<U8> data(8);
    data[0] = request_type;
    data[1] = request;
    write_le16(data.data() + 2, value);
    write_le16(data.data() + 4, index);
    write_le16(data.data() + 6, length);
    return data;
}

void setup(UsbModel &model, Usbd_Device *device, U8 request_type, U8 request,
           U16 value, U16 index, U16 length) {
    std::vector<U8> data = setup_bytes(request_type, request, value,
                                       index, length);
    model.out(device, 0, data.data(), (U16)data.size(), true);
}

std::vector<U8> control_in(UsbModel &model, Usbd_Device *device,
                           U8 request_type, U8 request, U16 value,
                           U16 index, U16 length) {
    setup(model, device, request_type, request, value, index, length);
    std::vector<U8> result;
    for (U32 packets = 0u; packets < 8u &&
         (model.endpoints[0] & USBD__EP_STAT_TX_MASK) ==
             USBD__EP_STAT_TX_VALID;
         packets++) {
        std::vector<U8> packet = model.in(device, 0);
        result.insert(result.end(), packet.begin(), packet.end());
    }
    CHECK((model.endpoints[0] & USBD__EP_STAT_TX_MASK) !=
          USBD__EP_STAT_TX_VALID);
    model.out(device, 0, nullptr, 0);
    return result;
}

void control_out(UsbModel &model, Usbd_Device *device, U8 request_type,
                 U8 request, U16 value, U16 index) {
    setup(model, device, request_type, request, value, index, 0);
    CHECK(model.in(device, 0).empty());
}

void control_expect_stall(UsbModel &model, Usbd_Device *device,
                          U8 request_type, U8 request, U16 value, U16 index,
                          U16 length) {
    setup(model, device, request_type, request, value, index, length);
    CHECK((model.endpoints[0] & USBD__EP_STAT_TX_MASK) ==
          USBD__EP_STAT_TX_STALL);
    CHECK((model.endpoints[0] & USBD__EP_STAT_RX_MASK) ==
          USBD__EP_STAT_RX_STALL);
}

} // namespace

ModelRegister::operator U32() const {
    (void)device_;
    return active_model->read(offset_);
}

ModelRegister &ModelRegister::operator=(U32 value) {
    (void)device_;
    active_model->write(offset_, value);
    return *this;
}

#endif

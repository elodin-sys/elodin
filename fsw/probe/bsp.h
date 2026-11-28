#ifndef BSP_H
#define BSP_H

// Aleph Flight Controller debug probe (U2, STM32G0B1KBU6N). See modules/Debugger/Debugger.zen.
//
//   PB0  SWCLK  to STM32H747 PA14   push-pull, direct (no translator)
//   PB1  SWDIO  to STM32H747 PA13   bidirectional, direct
//   PC14 NRST   to MCU_NRST         open drain (PC14 only sinks)
//   PB9  BOOT0  to MCU_BOOT0        released (input) unless forced high
//   PB8  INT    MCU_INT             input, unused
//   PA0  USART4_TX -> STM32 UART3 RX    PA1 USART4_RX <- STM32 UART3 TX
//   PD3  USART5_TX -> STM32 UART6 RX    PD2 USART5_RX <- STM32 UART6 TX
//   PA4  IND_LED, yellow-green, active high
//   PA5  STATUS_LED DIN (WL-ICLED), TIM2_CH1
//   PA7  VBUS_3V3 sense (unused)
//   PA11/PA12 USB D-/D+

#include "core.h"
#include "hal.h"

typedef struct {
    bool initialized;
    bool target_swd_prepared;
    bool usb_device_prepared;
} Bsp;

typedef enum {
    BspUsb_NoFlags             = 0,
    BspUsb_Prepared            = 1u << 0,
    BspUsb_HasDpDm             = 1u << 1,
    BspUsb_HasIrq              = 1u << 2,
    BspUsb_UsesHsi48Crs        = 1u << 3,
    BspUsb_SharesUcpdIrq       = 1u << 4,
    BspUsb_HasInternalDpPullup = 1u << 5,
    BspUsb_HasPma              = 1u << 6,
    BspUsb_Pma32Bit            = 1u << 7,
    BspUsb_HasVddio2UsbSupply  = 1u << 8,
} Bsp_UsbFlags;

typedef struct {
    char const *owner;
    U32 flags;
    Hal_PeripheralInfo peripheral;
    Hal_UsbClockInfo clock;
    Hal_Pin dp_pin;
    Hal_Pin dm_pin;
    UPtr reg_base;
    UPtr pma_base;
    U32 pma_bytes;
    U32 usb_hz;
    U8 endpoint_count;
    Hal_Irq irq;
} Bsp_UsbDevice;

typedef struct {
    char const *owner;
    UPtr data_gpio_base;
    UPtr reset_gpio_base;
    U8 swdio_pin;
    U8 swd_dir_pin;  // SwdNoDirPin: SWDIO is wired straight to the target
    U8 swclk_pin;
    U8 nreset_pin;
} Bsp_SwdPort;

typedef enum {
    BspUart_Console0 = 0,  // USART4 <-> STM32H747 UART3
    BspUart_Console1,      // USART5 <-> STM32H747 UART6
    BspUart_Count,
} Bsp_UartId;

typedef struct {
    char const *owner;
    UPtr base;
    U32 kernel_hz;
    Hal_Irq irq;
} Bsp_Uart;

typedef enum {
    BspSignal_LedIndicator = 0,
    BspSignal_TargetBoot0,
    BspSignal_Count,
} Bsp_SignalId;

typedef enum {
    BspSignal_NoFlags  = 0,
    BspSignal_Prepared = 1u << 0,
    BspSignal_Writable = 1u << 1,
    BspSignal_Output   = 1u << 2,
} Bsp_SignalFlags;

typedef struct {
    Bsp_SignalId id;
    char const *owner;
    U32 flags;
    Hal_Pin pin;
    bool active;
} Bsp_Signal;

typedef struct {
    UPtr tim_base;
    U32 tim_hz;
} Bsp_RgbLed;

bool bsp_init(void);
Bsp const *bsp(void);
bool bsp_target_swd_prepare(Bsp_SwdPort *out, char const *owner);
bool bsp_usb_device_prepare(Bsp_UsbDevice *out, char const *owner);
bool bsp_uart_prepare(Bsp_Uart *out, Bsp_UartId id, char const *owner);
bool bsp_rgb_led_prepare(Bsp_RgbLed *out, char const *owner);
bool bsp_signal_prepare(Bsp_Signal *out, Bsp_SignalId id, char const *owner);
bool bsp_signal_write(Bsp_Signal *signal, bool active);
void bsp_fault_safe_outputs(void);

#define BSP__REG32(addr) (*(volatile U32 *)(addr))

#define BSP__PERIPH_BASE 0x40000000UL
#define BSP__AHB_BASE    (BSP__PERIPH_BASE + 0x00020000UL)
#define BSP__IOPORT_BASE 0x50000000UL

#define BSP__RCC_BASE    (BSP__AHB_BASE + 0x00001000UL)
#define BSP__GPIOA_BASE  (BSP__IOPORT_BASE + 0x00000000UL)
#define BSP__GPIOB_BASE  (BSP__IOPORT_BASE + 0x00000400UL)
#define BSP__GPIOC_BASE  (BSP__IOPORT_BASE + 0x00000800UL)

#define BSP__RCC_IOPENR  BSP__REG32(BSP__RCC_BASE + 0x34)

#define BSP__GPIO_MODER(base)  BSP__REG32((base) + 0x00)
#define BSP__GPIO_OTYPER(base) BSP__REG32((base) + 0x04)
#define BSP__GPIO_PUPDR(base)  BSP__REG32((base) + 0x0C)
#define BSP__GPIO_BSRR(base)   BSP__REG32((base) + 0x18)

#define BSP__RCC_IOPENR_GPIOAEN (1u << 0)
#define BSP__RCC_IOPENR_GPIOBEN (1u << 1)
#define BSP__RCC_IOPENR_GPIOCEN (1u << 2)

#define BSP__GPIO_MODE_INPUT  0u
#define BSP__GPIO_MODE_OUTPUT 1u

#define BSP__LED_IND_PIN 4u   // PA4
#define BSP__RGB_PIN     5u   // PA5
#define BSP__SWCLK_PIN   0u   // PB0
#define BSP__SWDIO_PIN   1u   // PB1
#define BSP__BOOT0_PIN   9u   // PB9
#define BSP__NRESET_PIN  14u  // PC14

#define BSP__RGB_AF_TIM2 2u
#define BSP__USART4_AF   4u
#define BSP__USART5_AF   3u

#define BSP__USB_PMA_BASE       (BSP__PERIPH_BASE + 0x00009800UL)
#define BSP__USB_PMA_BYTES      2048u
#define BSP__USB_ENDPOINT_COUNT 8u

static Bsp bsp__state;

// PD2/PD3 (USART5) are UCPD2_CC2/DBCC2: release the dead-battery pull-down.
static Hal_UcpdDeadBatteryConfig const bsp__ucpd2_dead_battery = {
    .peripheral = HalPeripheral_Ucpd2,
    .owner = "bsp-ucpd2-dead-battery",
    .flags = HalUcpdDeadBattery_Default,
};

static Hal_PeripheralConfig const bsp__usb_device = {
    .peripheral = HalPeripheral_Usb,
    .owner = "bsp-usb-device",
    .flags = HalPeripheralConfig_Default | HalPeripheralConfig_ResetPulse,
};

static Hal_UsbClockConfig const bsp__usb_clock = {
    .owner = "bsp-usb-clock",
    .flags = HalUsbClock_Default,
    .source = HalUsbClock_Hsi48CrsUsbSof,
};

static Hal_PinConfig const bsp__usb_dm_pin = {
    .pin = HAL_PIN_A11,
    .owner = "bsp-usb-dm",
    .flags = HalPinConfig_Default,
    .mode = HalPinMode_Input,
    .speed = HalSpeed_High,
};

static Hal_PinConfig const bsp__usb_dp_pin = {
    .pin = HAL_PIN_A12,
    .owner = "bsp-usb-dp",
    .flags = HalPinConfig_Default,
    .mode = HalPinMode_Input,
    .speed = HalSpeed_High,
};

static Hal_PinConfig const bsp__led_ind_pin = {
    .pin = HAL_PIN(HalPort_A, BSP__LED_IND_PIN),
    .owner = "bsp-led-indicator",
    .flags = HalPinConfig_NoFlags,
    .mode = HalPinMode_Output,
    .output = HalOutput_PushPull,
    .speed = HalSpeed_Low,
    .initial_level = HalLevel_Low,
};

// Idles low between WL-ICLED frames; the TIM2 alternate function takes it over.
static Hal_PinConfig const bsp__rgb_pin = {
    .pin = HAL_PIN(HalPort_A, BSP__RGB_PIN),
    .owner = "bsp-rgb-led",
    .flags = HalPinConfig_NoFlags,
    .mode = HalPinMode_Output,
    .output = HalOutput_PushPull,
    .speed = HalSpeed_Medium,
    .initial_level = HalLevel_Low,
};

static Hal_PinConfig const bsp__swdio_pin = {
    .pin = HAL_PIN(HalPort_B, BSP__SWDIO_PIN),
    .owner = "bsp-target-swdio",
    .flags = HalPinConfig_NoFlags,
    .mode = HalPinMode_Input,
    .output = HalOutput_PushPull,
    .speed = HalSpeed_VeryHigh,
    .initial_level = HalLevel_High,
};

static Hal_PinConfig const bsp__swclk_pin = {
    .pin = HAL_PIN(HalPort_B, BSP__SWCLK_PIN),
    .owner = "bsp-target-swclk",
    .flags = HalPinConfig_NoFlags,
    .mode = HalPinMode_Output,
    .output = HalOutput_PushPull,
    .speed = HalSpeed_VeryHigh,
    .initial_level = HalLevel_Low,
};

static Hal_PinConfig const bsp__nreset_pin = {
    .pin = HAL_PIN(HalPort_C, BSP__NRESET_PIN),
    .owner = "bsp-target-nreset",
    .flags = HalPinConfig_NoFlags,
    .mode = HalPinMode_Output,
    .output = HalOutput_OpenDrain,
    .speed = HalSpeed_Low,
    .initial_level = HalLevel_High,
};

// Released: R3 on the STM32H747 side holds BOOT0 low. Forcing it is a
// push-pull high; releasing returns the pin to input.
static Hal_PinConfig const bsp__boot0_pin = {
    .pin = HAL_PIN(HalPort_B, BSP__BOOT0_PIN),
    .owner = "bsp-target-boot0",
    .flags = HalPinConfig_NoFlags,
    .mode = HalPinMode_Input,
    .output = HalOutput_PushPull,
    .speed = HalSpeed_Low,
    .initial_level = HalLevel_Low,
};

typedef struct {
    Hal_Peripheral peripheral;
    Hal_Pin tx;
    Hal_Pin rx;
    U8 alternate;
} Bsp__UartDesc;

static Bsp__UartDesc const bsp__uart_descs[BspUart_Count] = {
    [BspUart_Console0] = {
        .peripheral = HalPeripheral_Usart4,
        .tx = HAL_PIN_A0,
        .rx = HAL_PIN_A1,
        .alternate = BSP__USART4_AF,
    },
    [BspUart_Console1] = {
        .peripheral = HalPeripheral_Usart5,
        .tx = HAL_PIN_D3,
        .rx = HAL_PIN_D2,
        .alternate = BSP__USART5_AF,
    },
};

typedef struct {
    Bsp_SignalId id;
    Hal_PinConfig const *pin_config;
    U32 flags;
} Bsp__SignalDesc;

static Bsp__SignalDesc const bsp__signal_descs[BspSignal_Count] = {
    [BspSignal_LedIndicator] = {
        .id = BspSignal_LedIndicator,
        .pin_config = &bsp__led_ind_pin,
        .flags = BspSignal_Writable | BspSignal_Output,
    },
    [BspSignal_TargetBoot0] = {
        .id = BspSignal_TargetBoot0,
        .pin_config = &bsp__boot0_pin,
        .flags = BspSignal_Writable,
    },
};

static UPtr bsp__gpio_base(Hal_Port port) {
    switch (port) {
    case HalPort_A: return BSP__GPIOA_BASE;
    case HalPort_B: return BSP__GPIOB_BASE;
    case HalPort_C: return BSP__GPIOC_BASE;
    default:        return 0;
    }
}

static void bsp__raw_pin_mode(UPtr base, U32 pin, U32 mode) {
    U32 shift = pin * 2u;
    U32 mask = 3u << shift;
    BSP__GPIO_MODER(base) = (BSP__GPIO_MODER(base) & ~mask) | (mode << shift);
}

static void bsp__raw_pin_no_pull(UPtr base, U32 pin) {
    BSP__GPIO_PUPDR(base) &= ~(3u << (pin * 2u));
}

// Also the fault path: SWD and BOOT0 released, nRESET released, LEDs off.
void bsp_fault_safe_outputs(void) {
    BSP__RCC_IOPENR |= BSP__RCC_IOPENR_GPIOAEN |
                       BSP__RCC_IOPENR_GPIOBEN |
                       BSP__RCC_IOPENR_GPIOCEN;
    arch_barrier();
    (void)BSP__RCC_IOPENR;

    // Preload every output before changing its mode to avoid target glitches.
    BSP__GPIO_BSRR(BSP__GPIOA_BASE) = ((1u << BSP__LED_IND_PIN) | (1u << BSP__RGB_PIN)) << 16;
    BSP__GPIO_BSRR(BSP__GPIOB_BASE) = (1u << BSP__SWCLK_PIN) << 16;
    BSP__GPIO_BSRR(BSP__GPIOC_BASE) = 1u << BSP__NRESET_PIN;

    BSP__GPIO_OTYPER(BSP__GPIOA_BASE) &= ~((1u << BSP__LED_IND_PIN) | (1u << BSP__RGB_PIN));
    BSP__GPIO_OTYPER(BSP__GPIOB_BASE) &= ~(1u << BSP__SWCLK_PIN);
    BSP__GPIO_OTYPER(BSP__GPIOC_BASE) |= 1u << BSP__NRESET_PIN;

    bsp__raw_pin_no_pull(BSP__GPIOA_BASE, BSP__LED_IND_PIN);
    bsp__raw_pin_no_pull(BSP__GPIOA_BASE, BSP__RGB_PIN);
    bsp__raw_pin_no_pull(BSP__GPIOB_BASE, BSP__SWCLK_PIN);
    bsp__raw_pin_no_pull(BSP__GPIOB_BASE, BSP__SWDIO_PIN);
    bsp__raw_pin_no_pull(BSP__GPIOB_BASE, BSP__BOOT0_PIN);
    bsp__raw_pin_no_pull(BSP__GPIOC_BASE, BSP__NRESET_PIN);

    bsp__raw_pin_mode(BSP__GPIOA_BASE, BSP__LED_IND_PIN, BSP__GPIO_MODE_OUTPUT);
    bsp__raw_pin_mode(BSP__GPIOA_BASE, BSP__RGB_PIN, BSP__GPIO_MODE_OUTPUT);
    bsp__raw_pin_mode(BSP__GPIOB_BASE, BSP__SWCLK_PIN, BSP__GPIO_MODE_OUTPUT);
    bsp__raw_pin_mode(BSP__GPIOB_BASE, BSP__SWDIO_PIN, BSP__GPIO_MODE_INPUT);
    bsp__raw_pin_mode(BSP__GPIOB_BASE, BSP__BOOT0_PIN, BSP__GPIO_MODE_INPUT);
    bsp__raw_pin_mode(BSP__GPIOC_BASE, BSP__NRESET_PIN, BSP__GPIO_MODE_OUTPUT);
    arch_barrier();
}

bool bsp_init(void) {
    bsp_fault_safe_outputs();

    if (!hal_ucpd_dead_battery_release(&bsp__ucpd2_dead_battery)) return false;
    if (!hal_pin_configure(&bsp__swdio_pin)) return false;
    if (!hal_pin_configure(&bsp__swclk_pin)) return false;
    if (!hal_pin_configure(&bsp__nreset_pin)) return false;
    if (!hal_pin_configure(&bsp__boot0_pin)) return false;
    if (!hal_pin_configure(&bsp__led_ind_pin)) return false;
    if (!hal_pin_configure(&bsp__rgb_pin)) return false;

    bsp__state.initialized = true;
    return true;
}

Bsp const *bsp(void) {
    return &bsp__state;
}

static bool bsp__pin_claim(Hal_PinConfig const *base, char const *owner) {
    Hal_PinConfig pin = *base;
    pin.owner = owner;
    pin.flags = HalPinConfig_Default;
    return hal_pin_configure(&pin);
}

bool bsp_target_swd_prepare(Bsp_SwdPort *out, char const *owner) {
    if (!out || !owner) return false;
    if (!bsp__pin_claim(&bsp__swdio_pin, owner)) return false;
    if (!bsp__pin_claim(&bsp__swclk_pin, owner)) return false;
    if (!bsp__pin_claim(&bsp__nreset_pin, owner)) return false;

    *out = (Bsp_SwdPort){
        .owner = owner,
        .data_gpio_base = BSP__GPIOB_BASE,
        .reset_gpio_base = BSP__GPIOC_BASE,
        .swdio_pin = BSP__SWDIO_PIN,
        .swd_dir_pin = 0xFFu,
        .swclk_pin = BSP__SWCLK_PIN,
        .nreset_pin = BSP__NRESET_PIN,
    };
    bsp__state.target_swd_prepared = true;
    return true;
}

bool bsp_usb_device_prepare(Bsp_UsbDevice *out, char const *owner) {
    if (!out || !owner) return false;

    Hal_PeripheralConfig peripheral = bsp__usb_device;
    peripheral.owner = owner;
    Hal_UsbClockConfig clock = bsp__usb_clock;
    clock.owner = owner;

    Hal_UsbClockInfo clock_info = {0};
    Hal_PeripheralInfo peripheral_info = {0};
    if (!hal_usb_clock_prepare(&clock, &clock_info)) return false;
    if (!hal_peripheral_prepare(&peripheral, &peripheral_info)) return false;
    if (!bsp__pin_claim(&bsp__usb_dm_pin, owner)) return false;
    if (!bsp__pin_claim(&bsp__usb_dp_pin, owner)) return false;

    out->owner = owner;
    out->flags = BspUsb_Prepared |
                 BspUsb_HasDpDm |
                 BspUsb_UsesHsi48Crs |
                 BspUsb_SharesUcpdIrq |
                 BspUsb_HasInternalDpPullup |
                 BspUsb_HasPma |
                 BspUsb_Pma32Bit |
                 BspUsb_HasVddio2UsbSupply;
    if (peripheral_info.irq != HalIrq_None) out->flags |= BspUsb_HasIrq;
    out->peripheral = peripheral_info;
    out->clock = clock_info;
    out->dp_pin = bsp__usb_dp_pin.pin;
    out->dm_pin = bsp__usb_dm_pin.pin;
    out->reg_base = peripheral_info.base;
    out->pma_base = BSP__USB_PMA_BASE;
    out->pma_bytes = BSP__USB_PMA_BYTES;
    out->usb_hz = clock_info.usb_hz;
    out->endpoint_count = BSP__USB_ENDPOINT_COUNT;
    out->irq = peripheral_info.irq;

    bsp__state.usb_device_prepared = true;
    return true;
}

// The STM32H747 side may be unpowered or in reset; the RX pull-up keeps a
// floating line from reading as breaks.
bool bsp_uart_prepare(Bsp_Uart *out, Bsp_UartId id, char const *owner) {
    if (!out || !owner || id >= BspUart_Count) return false;
    Bsp__UartDesc const *desc = &bsp__uart_descs[id];

    Hal_PeripheralInfo info = {0};
    if (!hal_peripheral_prepare(&(Hal_PeripheralConfig){
            .peripheral = desc->peripheral,
            .owner = owner,
            .flags = HalPeripheralConfig_Default | HalPeripheralConfig_ResetPulse,
        }, &info)) return false;
    if (!hal_pin_configure(&(Hal_PinConfig){
            .pin = desc->tx,
            .owner = owner,
            .flags = HalPinConfig_Default,
            .mode = HalPinMode_Alternate,
            .output = HalOutput_PushPull,
            .speed = HalSpeed_Medium,
            .alternate = desc->alternate,
            .initial_level = HalLevel_High,
        })) return false;
    if (!hal_pin_configure(&(Hal_PinConfig){
            .pin = desc->rx,
            .owner = owner,
            .flags = HalPinConfig_Default,
            .mode = HalPinMode_Alternate,
            .pull = HalPull_Up,
            .alternate = desc->alternate,
            .initial_level = HalLevel_Unchanged,
        })) return false;

    *out = (Bsp_Uart){
        .owner = owner,
        .base = info.base,
        .kernel_hz = info.bus_hz,
        .irq = info.irq,
    };
    return true;
}

bool bsp_rgb_led_prepare(Bsp_RgbLed *out, char const *owner) {
    if (!out || !owner) return false;

    Hal_PeripheralInfo info = {0};
    if (!hal_peripheral_prepare(&(Hal_PeripheralConfig){
            .peripheral = HalPeripheral_Tim2,
            .owner = owner,
            .flags = HalPeripheralConfig_Default | HalPeripheralConfig_ResetPulse,
        }, &info)) return false;
    if (!hal_pin_configure(&(Hal_PinConfig){
            .pin = HAL_PIN(HalPort_A, BSP__RGB_PIN),
            .owner = owner,
            .flags = HalPinConfig_Default,
            .mode = HalPinMode_Alternate,
            .output = HalOutput_PushPull,
            .speed = HalSpeed_Medium,
            .alternate = BSP__RGB_AF_TIM2,
            .initial_level = HalLevel_Low,
        })) return false;

    *out = (Bsp_RgbLed){ .tim_base = info.base, .tim_hz = info.bus_hz };
    return true;
}

bool bsp_signal_prepare(Bsp_Signal *out, Bsp_SignalId id, char const *owner) {
    if (!out || id >= BspSignal_Count) return false;
    Bsp__SignalDesc const *desc = &bsp__signal_descs[id];
    if (!bsp__pin_claim(desc->pin_config, owner)) return false;

    *out = (Bsp_Signal){
        .id = id,
        .owner = owner,
        .flags = desc->flags | BspSignal_Prepared,
        .pin = desc->pin_config->pin,
        .active = false,
    };
    return true;
}

bool bsp_signal_write(Bsp_Signal *signal, bool active) {
    if (!signal) return false;
    if ((signal->flags & (BspSignal_Prepared | BspSignal_Writable)) !=
        (BspSignal_Prepared | BspSignal_Writable)) return false;

    UPtr base = bsp__gpio_base(signal->pin.port);
    if (!base) return false;
    BSP__GPIO_BSRR(base) = 1u << (signal->pin.index + (active ? 0u : 16u));
    // BOOT0 is driven only while forced; released it floats onto R3.
    if ((signal->flags & BspSignal_Output) == 0u) {
        bsp__raw_pin_mode(base, signal->pin.index, active ? BSP__GPIO_MODE_OUTPUT : BSP__GPIO_MODE_INPUT);
    }
    signal->active = active;
    return true;
}

#endif

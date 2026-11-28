#ifndef HAL_H
#define HAL_H

#include "core.h"
#include "arch.h"

typedef enum {
    HalCore_NoFlags = 0,
} Hal_CoreFlags;

typedef enum {
    HalClock_Hsi16_16MHz = 0,
    HalClock_Hsi16Pll_64MHz,
} Hal_ClockProfile;

typedef struct {
    Hal_ClockProfile clock;
    U32 flags;
} Hal_CoreConfig;

typedef struct {
    Hal_ClockProfile clock;
    U32 flags;
    U32 sys_hz;
    U32 ahb_hz;
    U32 apb_hz;
    U32 systick_hz;
    U32 systick_cycles;
    volatile U32 tick_ms;
    bool systick_enabled;
} Hal_Core;

typedef enum {
    HalPeripheral_None = 0,
    HalPeripheral_GpioA,
    HalPeripheral_GpioB,
    HalPeripheral_GpioC,
    HalPeripheral_GpioD,
    HalPeripheral_GpioF,
    HalPeripheral_Syscfg,
    HalPeripheral_Pwr,
    HalPeripheral_Crs,
    HalPeripheral_Tim2,
    HalPeripheral_Usart1,
    HalPeripheral_Usart2,
    HalPeripheral_Usart4,
    HalPeripheral_Usart5,
    HalPeripheral_Usb,
    HalPeripheral_Spi1,
    HalPeripheral_I2c1,
    HalPeripheral_Ucpd1,
    HalPeripheral_Ucpd2,
    HalPeripheral_Count,
} Hal_Peripheral;

typedef enum {
    HalIrq_UsbUcpd1_2 = 8,
    HalIrq_Dma1Channel1 = 9,
    HalIrq_Dma1Channel2_3 = 10,
    HalIrq_I2c1 = 23,
    HalIrq_Spi1 = 25,
    HalIrq_Usart1 = 27,
    HalIrq_Usart2Lpuart2 = 28,
    HalIrq_Usart3_4_5_6Lpuart1 = 29,

    HalIrq_None = ARCH_IRQ_NONE,
} Hal_Irq;

typedef enum {
    HalPeripheralInfo_NoFlags = 0,
    HalPeripheralInfo_Prepared = 1u << 0,
    HalPeripheralInfo_Clocked  = 1u << 1,
    HalPeripheralInfo_Reset    = 1u << 2,
} Hal_PeripheralInfoFlags;

typedef struct {
    Hal_Peripheral peripheral;
    U32 flags;
    UPtr base;
    U32 bus_hz;
    Hal_Irq irq;
} Hal_PeripheralInfo;

typedef enum {
    HalPeripheralConfig_NoFlags    = 0,
    HalPeripheralConfig_Reserve    = 1u << 0,
    HalPeripheralConfig_Clock      = 1u << 1,
    HalPeripheralConfig_ResetPulse = 1u << 2,

    HalPeripheralConfig_Default = HalPeripheralConfig_Reserve | HalPeripheralConfig_Clock,
} Hal_PeripheralConfigFlags;

typedef struct {
    Hal_Peripheral peripheral;
    char const *owner;
    U32 flags;
} Hal_PeripheralConfig;

typedef enum {
    HalUsbClock_Hsi48CrsUsbSof = 0,
} Hal_UsbClockSource;

typedef enum {
    HalUsbClock_NoFlags       = 0,
    HalUsbClock_Reserve       = 1u << 0,
    HalUsbClock_EnableVddio2  = 1u << 1,
    HalUsbClock_EnableCrs     = 1u << 2,

    HalUsbClock_Default = HalUsbClock_Reserve |
                          HalUsbClock_EnableVddio2 |
                          HalUsbClock_EnableCrs,
} Hal_UsbClockFlags;

typedef struct {
    char const *owner;
    U32 flags;
    Hal_UsbClockSource source;
} Hal_UsbClockConfig;

typedef enum {
    HalUsbClockInfo_NoFlags       = 0,
    HalUsbClockInfo_Prepared      = 1u << 0,
    HalUsbClockInfo_Hsi48Ready    = 1u << 1,
    HalUsbClockInfo_Vddio2Enabled = 1u << 2,
    HalUsbClockInfo_CrsConfigured = 1u << 3,
} Hal_UsbClockInfoFlags;

typedef struct {
    U32 flags;
    U32 usb_hz;
    U32 hsi48_hz;
    UPtr crs_base;
    UPtr rcc_base;
} Hal_UsbClockInfo;

typedef enum {
    HalPort_A = 0,
    HalPort_B,
    HalPort_C,
    HalPort_D,
    HalPort_F,
    HalPort_Count,
} Hal_Port;

typedef struct {
    Hal_Port port;
    U8 index;
} Hal_Pin;

#define HAL_PIN(port_, index_) ((Hal_Pin){ .port = (port_), .index = (index_) })
#define HAL_PIN_A0 HAL_PIN(HalPort_A, 0)
#define HAL_PIN_A1 HAL_PIN(HalPort_A, 1)
#define HAL_PIN_A2 HAL_PIN(HalPort_A, 2)
#define HAL_PIN_A3 HAL_PIN(HalPort_A, 3)
#define HAL_PIN_A11 HAL_PIN(HalPort_A, 11)
#define HAL_PIN_A12 HAL_PIN(HalPort_A, 12)
#define HAL_PIN_D0 HAL_PIN(HalPort_D, 0)
#define HAL_PIN_D1 HAL_PIN(HalPort_D, 1)
#define HAL_PIN_D2 HAL_PIN(HalPort_D, 2)
#define HAL_PIN_D3 HAL_PIN(HalPort_D, 3)

typedef enum {
    HalPinMode_Input = 0,
    HalPinMode_Output,
    HalPinMode_Analog,
    HalPinMode_Alternate,
} Hal_PinMode;

typedef enum {
    HalPull_None = 0,
    HalPull_Up,
    HalPull_Down,
} Hal_Pull;

typedef enum {
    HalOutput_PushPull = 0,
    HalOutput_OpenDrain,
} Hal_OutputType;

typedef enum {
    HalSpeed_Low = 0,
    HalSpeed_Medium,
    HalSpeed_High,
    HalSpeed_VeryHigh,
} Hal_PinSpeed;

typedef enum {
    HalLevel_Low = 0,
    HalLevel_High,
    HalLevel_Unchanged,
} Hal_Level;

typedef enum {
    HalPinConfig_NoFlags = 0,
    HalPinConfig_Reserve = 1u << 0,

    HalPinConfig_Default = HalPinConfig_Reserve,
} Hal_PinConfigFlags;

typedef struct {
    Hal_Pin pin;
    char const *owner;
    U32 flags;

    Hal_PinMode mode;
    Hal_Pull pull;
    Hal_OutputType output;
    Hal_PinSpeed speed;

    U8 alternate;
    Hal_Level initial_level;
} Hal_PinConfig;

typedef enum {
    HalUcpdDeadBattery_NoFlags = 0,
    HalUcpdDeadBattery_Reserve = 1u << 0,

    HalUcpdDeadBattery_Default = HalUcpdDeadBattery_Reserve,
} Hal_UcpdDeadBatteryFlags;

typedef struct {
    Hal_Peripheral peripheral;
    char const *owner;
    U32 flags;
} Hal_UcpdDeadBatteryConfig;

void hal_core_init(Hal_CoreConfig const *config);
Hal_Core const *hal_core(void);
U32 hal_now_us(void);
void hal_delay_us(U32 us);
// Independent watchdog: LSI-clocked, cannot be stopped once started, frozen
// while a debugger halts the core. timeout_ms up to 8190.
void hal_watchdog_start(U32 timeout_ms);
void hal_watchdog_feed(void);
bool hal_time_expired_us(U32 start_us, U32 duration_us);
void hal_system_reset(void);
void hal_wait_for_interrupt(void);
bool hal_peripheral_prepare(Hal_PeripheralConfig const *config, Hal_PeripheralInfo *out);
bool hal_pin_configure(Hal_PinConfig const *config);
bool hal_ucpd_dead_battery_release(Hal_UcpdDeadBatteryConfig const *config);
bool hal_usb_clock_prepare(Hal_UsbClockConfig const *config, Hal_UsbClockInfo *out);
void SysTick_Handler(void);

#define HAL__REG32(addr) (*(volatile U32 *)(addr))

#define HAL__PERIPH_BASE 0x40000000UL
#define HAL__APB_BASE    HAL__PERIPH_BASE
#define HAL__AHB_BASE    (HAL__PERIPH_BASE + 0x00020000UL)
#define HAL__IOPORT_BASE 0x50000000UL

#define HAL__RCC_BASE    (HAL__AHB_BASE + 0x00001000UL)
#define HAL__FLASH_BASE  (HAL__AHB_BASE + 0x00002000UL)
#define HAL__SYSCFG_BASE (HAL__APB_BASE + 0x00010000UL)
#define HAL__CRS_BASE    (HAL__APB_BASE + 0x00006C00UL)
#define HAL__PWR_BASE    (HAL__APB_BASE + 0x00007000UL)

#define HAL__GPIOA_BASE  (HAL__IOPORT_BASE + 0x00000000UL)
#define HAL__GPIOB_BASE  (HAL__IOPORT_BASE + 0x00000400UL)
#define HAL__GPIOC_BASE  (HAL__IOPORT_BASE + 0x00000800UL)
#define HAL__GPIOD_BASE  (HAL__IOPORT_BASE + 0x00000C00UL)
#define HAL__GPIOF_BASE  (HAL__IOPORT_BASE + 0x00001400UL)
#define HAL__USB_BASE    (HAL__APB_BASE + 0x00005C00UL)
#define HAL__USB_PMA_BASE (HAL__APB_BASE + 0x00009800UL)

#define HAL__RCC_CR_ADDR       (HAL__RCC_BASE + 0x00)
#define HAL__RCC_CFGR_ADDR     (HAL__RCC_BASE + 0x08)
#define HAL__RCC_PLLCFGR_ADDR  (HAL__RCC_BASE + 0x0C)
#define HAL__RCC_IOPRSTR_ADDR  (HAL__RCC_BASE + 0x24)
#define HAL__RCC_APBRSTR1_ADDR (HAL__RCC_BASE + 0x28)
#define HAL__RCC_APBRSTR2_ADDR (HAL__RCC_BASE + 0x2C)
#define HAL__RCC_IOPENR_ADDR   (HAL__RCC_BASE + 0x34)
#define HAL__RCC_APBENR1_ADDR  (HAL__RCC_BASE + 0x3C)
#define HAL__RCC_APBENR2_ADDR  (HAL__RCC_BASE + 0x40)
#define HAL__RCC_CCIPR_ADDR    (HAL__RCC_BASE + 0x54)

#define HAL__RCC_CR       HAL__REG32(HAL__RCC_CR_ADDR)
#define HAL__RCC_CFGR     HAL__REG32(HAL__RCC_CFGR_ADDR)
#define HAL__RCC_PLLCFGR  HAL__REG32(HAL__RCC_PLLCFGR_ADDR)
#define HAL__RCC_IOPRSTR  HAL__REG32(HAL__RCC_IOPRSTR_ADDR)
#define HAL__RCC_APBRSTR1 HAL__REG32(HAL__RCC_APBRSTR1_ADDR)
#define HAL__RCC_APBRSTR2 HAL__REG32(HAL__RCC_APBRSTR2_ADDR)
#define HAL__RCC_IOPENR   HAL__REG32(HAL__RCC_IOPENR_ADDR)
#define HAL__RCC_APBENR1  HAL__REG32(HAL__RCC_APBENR1_ADDR)
#define HAL__RCC_APBENR2  HAL__REG32(HAL__RCC_APBENR2_ADDR)
#define HAL__RCC_CCIPR    HAL__REG32(HAL__RCC_CCIPR_ADDR)

#define HAL__FLASH_ACR   HAL__REG32(HAL__FLASH_BASE + 0x00)

#define HAL__SYSCFG_CFGR1 HAL__REG32(HAL__SYSCFG_BASE + 0x00)

#define HAL__PWR_CR2 HAL__REG32(HAL__PWR_BASE + 0x04)

#define HAL__CRS_CR   HAL__REG32(HAL__CRS_BASE + 0x00)
#define HAL__CRS_CFGR HAL__REG32(HAL__CRS_BASE + 0x04)
#define HAL__CRS_ICR  HAL__REG32(HAL__CRS_BASE + 0x0C)

#define HAL__GPIO_MODER(base)   HAL__REG32((base) + 0x00)
#define HAL__GPIO_OTYPER(base)  HAL__REG32((base) + 0x04)
#define HAL__GPIO_OSPEEDR(base) HAL__REG32((base) + 0x08)
#define HAL__GPIO_PUPDR(base)   HAL__REG32((base) + 0x0C)
#define HAL__GPIO_BSRR(base)    HAL__REG32((base) + 0x18)
#define HAL__GPIO_AFRL(base)    HAL__REG32((base) + 0x20)
#define HAL__GPIO_AFRH(base)    HAL__REG32((base) + 0x24)

#define HAL__BITS(width_) ((1u << (width_)) - 1u)
#define HAL__PIN_FIELD(pin_, width_, value_) ((U32)(value_) << ((pin_) * (width_)))
#define HAL__PIN_MASK(pin_, width_) HAL__PIN_FIELD((pin_), (width_), HAL__BITS(width_))
#define HAL__REG_FIELD_WRITE(reg_, mask_, value_) ((reg_) = ((reg_) & ~(mask_)) | (value_))
#define HAL__PIN_FIELD_WRITE(reg_, pin_, width_, value_) \
    HAL__REG_FIELD_WRITE((reg_), HAL__PIN_MASK((pin_), (width_)), HAL__PIN_FIELD((pin_), (width_), (value_)))

#define HAL__RCC_CR_HSION  (1u << 8)
#define HAL__RCC_CR_HSIRDY (1u << 10)
#define HAL__RCC_CR_HSI48ON  (1u << 22)
#define HAL__RCC_CR_HSI48RDY (1u << 23)
#define HAL__RCC_CR_PLLON  (1u << 24)
#define HAL__RCC_CR_PLLRDY (1u << 25)

#define HAL__RCC_CFGR_SW_MASK     (7u << 0)
#define HAL__RCC_CFGR_SW_HSI16    (0u << 0)
#define HAL__RCC_CFGR_SW_PLLRCLK  (2u << 0)
#define HAL__RCC_CFGR_SWS_MASK    (7u << 3)
#define HAL__RCC_CFGR_SWS_HSI16   (0u << 3)
#define HAL__RCC_CFGR_SWS_PLLRCLK (2u << 3)

#define HAL__RCC_PLLCFGR_PLLSRC_HSI16 (2u << 0)
#define HAL__RCC_PLLCFGR_PLLM_DIV1    (0u << 4)
#define HAL__RCC_PLLCFGR_PLLN_MUL8    (8u << 8)
#define HAL__RCC_PLLCFGR_PLLREN       (1u << 28)
#define HAL__RCC_PLLCFGR_PLLR_DIV2    (1u << 29)

#define HAL__RCC_IOPENR_GPIOAEN (1u << 0)
#define HAL__RCC_IOPENR_GPIOBEN (1u << 1)
#define HAL__RCC_IOPENR_GPIOCEN (1u << 2)
#define HAL__RCC_IOPENR_GPIODEN (1u << 3)
#define HAL__RCC_IOPENR_GPIOFEN (1u << 5)

#define HAL__RCC_APBENR2_SYSCFGEN      (1u << 0)
#define HAL__RCC_APBENR1_PWREN         (1u << 28)
#define HAL__RCC_APBENR1_DBGEN         (1u << 27)

#define HAL__IWDG_KR   HAL__REG32(0x40003000UL + 0x00)
#define HAL__IWDG_PR   HAL__REG32(0x40003000UL + 0x04)
#define HAL__IWDG_RLR  HAL__REG32(0x40003000UL + 0x08)
#define HAL__IWDG_SR   HAL__REG32(0x40003000UL + 0x0C)
#define HAL__DBG_APB_FZ1 HAL__REG32(0x40015800UL + 0x08)
#define HAL__DBG_IWDG_STOP (1u << 12)
#define HAL__RCC_APBENR1_CRSEN         (1u << 16)
#define HAL__RCC_APBENR1_TIM2EN        (1u << 0)
#define HAL__RCC_APBENR2_USART1EN      (1u << 14)
#define HAL__RCC_APBENR1_USART2EN      (1u << 17)
#define HAL__RCC_APBENR1_USART4EN      (1u << 19)
#define HAL__RCC_APBENR1_USART5EN      (1u << 8)
#define HAL__RCC_APBENR1_USBEN         (1u << 13)
#define HAL__RCC_APBENR2_SPI1EN        (1u << 12)
#define HAL__RCC_APBENR1_I2C1EN        (1u << 21)
#define HAL__RCC_APBENR1_UCPD1EN       (1u << 25)
#define HAL__RCC_APBENR1_UCPD2EN       (1u << 26)

#define HAL__RCC_CCIPR_USBSEL_MASK  (3u << 12)
#define HAL__RCC_CCIPR_USBSEL_HSI48 (0u << 12)

#define HAL__SYSCFG_CFGR1_UCPD1_STROBE (1u << 9)
#define HAL__SYSCFG_CFGR1_UCPD2_STROBE (1u << 10)

#define HAL__PWR_CR2_PVM_VDDIO2_MASK   (7u << 8)
#define HAL__PWR_CR2_PVM_VDDIO2_BYPASS (6u << 8)

#define HAL__CRS_CR_CEN         (1u << 5)
#define HAL__CRS_CR_AUTOTRIMEN  (1u << 6)
#define HAL__CRS_CR_TRIM_MASK   (0x7Fu << 8)
#define HAL__CRS_CFGR_SYNCSRC_USB_SOF (2u << 28)
#define HAL__CRS_CFGR_FELIM(value_)   ((U32)(value_) << 16)
#define HAL__CRS_CFGR_RELOAD(value_)  ((U32)(value_) << 0)
#define HAL__CRS_ICR_ALL_FLAGS        0x0Fu

#define HAL__FLASH_ACR_LATENCY_MASK (7u << 0)
#define HAL__FLASH_ACR_LATENCY_0WS  (0u << 0)
#define HAL__FLASH_ACR_LATENCY_2WS  (2u << 0)
#define HAL__FLASH_ACR_PRFTEN       (1u << 8)

#define HAL__HSI16_HZ 16000000u
#define HAL__HSI48_HZ 48000000u
#define HAL__PLL64_HZ 64000000u
#define HAL__SYSTICK_HZ 1000u
#define HAL__USB_PMA_BYTES 2048u

typedef struct {
    bool used;
    char const *owner;
} Hal__Reservation;

typedef struct {
    Hal_Peripheral peripheral;
    UPtr base;
    UPtr clock_reg;
    U32 clock_bit;
    UPtr reset_reg;
    U32 reset_bit;
    Hal_Irq irq;
} Hal__PeripheralDesc;

static Hal_Core hal__core;
static Hal__Reservation hal__peripheral_reservations[HalPeripheral_Count];
static Hal__Reservation hal__pin_reservations[HalPort_Count][16];
static Hal__Reservation hal__ucpd_dead_battery_reservations[HalPeripheral_Count];
static Hal__Reservation hal__usb_clock_reservation;

#define HAL__PERIPH_DESC(peripheral_, base_, clock_reg_, clock_bit_, reset_reg_, reset_bit_, irq_) \
    [peripheral_] = { (peripheral_), (base_), (clock_reg_), (clock_bit_), (reset_reg_), (reset_bit_), (irq_) }
#define HAL__GPIO_DESC(peripheral_, base_, bit_) \
    HAL__PERIPH_DESC((peripheral_), (base_), HAL__RCC_IOPENR_ADDR, (bit_), HAL__RCC_IOPRSTR_ADDR, (bit_), HalIrq_None)
#define HAL__APB1_DESC(peripheral_, base_, bit_, irq_) \
    HAL__PERIPH_DESC((peripheral_), (base_), HAL__RCC_APBENR1_ADDR, (bit_), HAL__RCC_APBRSTR1_ADDR, (bit_), (irq_))
#define HAL__APB2_DESC(peripheral_, base_, bit_, irq_) \
    HAL__PERIPH_DESC((peripheral_), (base_), HAL__RCC_APBENR2_ADDR, (bit_), HAL__RCC_APBRSTR2_ADDR, (bit_), (irq_))

static Hal__PeripheralDesc const hal__peripheral_descs[] = {
    HAL__GPIO_DESC(HalPeripheral_GpioA, HAL__GPIOA_BASE, HAL__RCC_IOPENR_GPIOAEN),
    HAL__GPIO_DESC(HalPeripheral_GpioB, HAL__GPIOB_BASE, HAL__RCC_IOPENR_GPIOBEN),
    HAL__GPIO_DESC(HalPeripheral_GpioC, HAL__GPIOC_BASE, HAL__RCC_IOPENR_GPIOCEN),
    HAL__GPIO_DESC(HalPeripheral_GpioD, HAL__GPIOD_BASE, HAL__RCC_IOPENR_GPIODEN),
    HAL__GPIO_DESC(HalPeripheral_GpioF, HAL__GPIOF_BASE, HAL__RCC_IOPENR_GPIOFEN),
    HAL__APB2_DESC(HalPeripheral_Syscfg, HAL__SYSCFG_BASE,               HAL__RCC_APBENR2_SYSCFGEN, HalIrq_None),
    HAL__APB1_DESC(HalPeripheral_Pwr,    HAL__PWR_BASE,                  HAL__RCC_APBENR1_PWREN, HalIrq_None),
    HAL__APB1_DESC(HalPeripheral_Crs,    HAL__CRS_BASE,                  HAL__RCC_APBENR1_CRSEN, HalIrq_None),
    HAL__APB1_DESC(HalPeripheral_Tim2,   HAL__APB_BASE + 0x00000000UL, HAL__RCC_APBENR1_TIM2EN, HalIrq_None),
    HAL__APB2_DESC(HalPeripheral_Usart1, HAL__APB_BASE + 0x00013800UL, HAL__RCC_APBENR2_USART1EN, HalIrq_Usart1),
    HAL__APB1_DESC(HalPeripheral_Usart2, HAL__APB_BASE + 0x00004400UL, HAL__RCC_APBENR1_USART2EN, HalIrq_Usart2Lpuart2),
    HAL__APB1_DESC(HalPeripheral_Usart4, HAL__APB_BASE + 0x00004C00UL, HAL__RCC_APBENR1_USART4EN, HalIrq_Usart3_4_5_6Lpuart1),
    HAL__APB1_DESC(HalPeripheral_Usart5, HAL__APB_BASE + 0x00005000UL, HAL__RCC_APBENR1_USART5EN, HalIrq_Usart3_4_5_6Lpuart1),
    HAL__APB1_DESC(HalPeripheral_Usb,    HAL__USB_BASE,                  HAL__RCC_APBENR1_USBEN, HalIrq_UsbUcpd1_2),
    HAL__APB2_DESC(HalPeripheral_Spi1,   HAL__APB_BASE + 0x00013000UL, HAL__RCC_APBENR2_SPI1EN, HalIrq_Spi1),
    HAL__APB1_DESC(HalPeripheral_I2c1,   HAL__APB_BASE + 0x00005400UL, HAL__RCC_APBENR1_I2C1EN, HalIrq_I2c1),
    HAL__APB1_DESC(HalPeripheral_Ucpd1,  HAL__APB_BASE + 0x0000A000UL, HAL__RCC_APBENR1_UCPD1EN, HalIrq_UsbUcpd1_2),
    HAL__APB1_DESC(HalPeripheral_Ucpd2,  HAL__APB_BASE + 0x0000A400UL, HAL__RCC_APBENR1_UCPD2EN, HalIrq_UsbUcpd1_2),
};

static bool hal__valid_peripheral(Hal_Peripheral peripheral) {
    return peripheral > HalPeripheral_None && peripheral < HalPeripheral_Count;
}

static bool hal__valid_pin(Hal_Pin pin) {
    return pin.port < HalPort_Count && pin.index < 16u;
}

static bool hal__reserve(Hal__Reservation *reservation, char const *owner) {
    if (!reservation->used) {
        reservation->used = true;
        reservation->owner = owner;
        return true;
    }

    return reservation->owner == owner;
}

static Hal__PeripheralDesc const *hal__peripheral_desc(Hal_Peripheral peripheral) {
    if (!hal__valid_peripheral(peripheral)) return 0;
    Hal__PeripheralDesc const *desc = &hal__peripheral_descs[peripheral];
    if (desc->peripheral != peripheral) return 0;
    return desc;
}

static Hal_Peripheral hal__gpio_peripheral_for_port(Hal_Port port) {
    switch (port) {
    case HalPort_A: return HalPeripheral_GpioA;
    case HalPort_B: return HalPeripheral_GpioB;
    case HalPort_C: return HalPeripheral_GpioC;
    case HalPort_D: return HalPeripheral_GpioD;
    case HalPort_F: return HalPeripheral_GpioF;
    default:        return HalPeripheral_None;
    }
}

static bool hal__peripheral_clock_enable(Hal_Peripheral peripheral) {
    Hal__PeripheralDesc const *desc = hal__peripheral_desc(peripheral);
    if (!desc || !desc->clock_reg || !desc->clock_bit) return false;

    HAL__REG32(desc->clock_reg) |= desc->clock_bit;
    arch_barrier();
    (void)HAL__REG32(desc->clock_reg);
    return true;
}

static bool hal__peripheral_reset_pulse(Hal_Peripheral peripheral) {
    Hal__PeripheralDesc const *desc = hal__peripheral_desc(peripheral);
    if (!desc || !desc->reset_reg || !desc->reset_bit) return false;

    HAL__REG32(desc->reset_reg) |= desc->reset_bit;
    arch_barrier();
    HAL__REG32(desc->reset_reg) &= ~desc->reset_bit;
    arch_barrier();
    return true;
}

static UPtr hal__gpio_base(Hal_Port port) {
    Hal__PeripheralDesc const *desc = hal__peripheral_desc(hal__gpio_peripheral_for_port(port));
    if (!desc) return 0;
    return desc->base;
}

static U32 hal__peripheral_bus_hz(Hal_Peripheral peripheral) {
    switch (peripheral) {
    case HalPeripheral_GpioA:
    case HalPeripheral_GpioB:
    case HalPeripheral_GpioC:
    case HalPeripheral_GpioD:
    case HalPeripheral_GpioF:
        return hal__core.ahb_hz;

    default:
        return hal__core.apb_hz;
    }
}

static U32 hal__pin_mode_bits(Hal_PinMode mode) {
    switch (mode) {
    case HalPinMode_Output:    return 1u;
    case HalPinMode_Alternate: return 2u;
    case HalPinMode_Analog:    return 3u;
    case HalPinMode_Input:
    default:                   return 0u;
    }
}

static U32 hal__pin_pull_bits(Hal_Pull pull) {
    switch (pull) {
    case HalPull_Up:   return 1u;
    case HalPull_Down: return 2u;
    case HalPull_None:
    default:           return 0u;
    }
}

static U32 hal__pin_speed_bits(Hal_PinSpeed speed) {
    switch (speed) {
    case HalSpeed_Medium:   return 1u;
    case HalSpeed_High:     return 2u;
    case HalSpeed_VeryHigh: return 3u;
    case HalSpeed_Low:
    default:                return 0u;
    }
}

static void hal__clock_hsi16(void) {
    HAL__RCC_CR |= HAL__RCC_CR_HSION;
    while ((HAL__RCC_CR & HAL__RCC_CR_HSIRDY) == 0) {
    }

    HAL__RCC_CFGR = (HAL__RCC_CFGR & ~HAL__RCC_CFGR_SW_MASK) | HAL__RCC_CFGR_SW_HSI16;
    while ((HAL__RCC_CFGR & HAL__RCC_CFGR_SWS_MASK) != HAL__RCC_CFGR_SWS_HSI16) {
    }

    HAL__FLASH_ACR = (HAL__FLASH_ACR & ~HAL__FLASH_ACR_LATENCY_MASK) | HAL__FLASH_ACR_LATENCY_0WS;
}

static void hal__clock_hsi16_pll64(void) {
    HAL__RCC_CR |= HAL__RCC_CR_HSION;
    while ((HAL__RCC_CR & HAL__RCC_CR_HSIRDY) == 0) {
    }

    HAL__RCC_CFGR = (HAL__RCC_CFGR & ~HAL__RCC_CFGR_SW_MASK) | HAL__RCC_CFGR_SW_HSI16;
    while ((HAL__RCC_CFGR & HAL__RCC_CFGR_SWS_MASK) != HAL__RCC_CFGR_SWS_HSI16) {
    }

    HAL__RCC_CR &= ~HAL__RCC_CR_PLLON;
    while ((HAL__RCC_CR & HAL__RCC_CR_PLLRDY) != 0) {
    }

    HAL__FLASH_ACR = (HAL__FLASH_ACR & ~HAL__FLASH_ACR_LATENCY_MASK) | HAL__FLASH_ACR_LATENCY_2WS | HAL__FLASH_ACR_PRFTEN;
    HAL__RCC_PLLCFGR = HAL__RCC_PLLCFGR_PLLSRC_HSI16 |
                       HAL__RCC_PLLCFGR_PLLM_DIV1 |
                       HAL__RCC_PLLCFGR_PLLN_MUL8 |
                       HAL__RCC_PLLCFGR_PLLREN |
                       HAL__RCC_PLLCFGR_PLLR_DIV2;

    HAL__RCC_CR |= HAL__RCC_CR_PLLON;
    while ((HAL__RCC_CR & HAL__RCC_CR_PLLRDY) == 0) {
    }

    HAL__RCC_CFGR = (HAL__RCC_CFGR & ~HAL__RCC_CFGR_SW_MASK) | HAL__RCC_CFGR_SW_PLLRCLK;
    while ((HAL__RCC_CFGR & HAL__RCC_CFGR_SWS_MASK) != HAL__RCC_CFGR_SWS_PLLRCLK) {
    }
}

static U32 hal__clock_init(Hal_ClockProfile clock) {
    switch (clock) {
    case HalClock_Hsi16Pll_64MHz:
        hal__clock_hsi16_pll64();
        return HAL__PLL64_HZ;

    case HalClock_Hsi16_16MHz:
    default:
        hal__clock_hsi16();
        return HAL__HSI16_HZ;
    }
}

static void hal__systick_init(U32 sys_hz) {
    U32 cycles = HAL__HSI16_HZ / HAL__SYSTICK_HZ;
    if (sys_hz == HAL__PLL64_HZ) cycles = HAL__PLL64_HZ / HAL__SYSTICK_HZ;

    ARCH__SYSTICK_CSR = 0;
    ARCH__SYSTICK_RVR = cycles - 1u;
    ARCH__SYSTICK_CVR = 0;

    hal__core.systick_hz = HAL__SYSTICK_HZ;
    hal__core.systick_cycles = cycles;
    hal__core.tick_ms = 0;
    hal__core.systick_enabled = true;

    ARCH__SYSTICK_CSR = ARCH__SYSTICK_CSR_CLKSOURCE |
                        ARCH__SYSTICK_CSR_TICKINT |
                        ARCH__SYSTICK_CSR_ENABLE;
}

void hal_core_init(Hal_CoreConfig const *config) {
    Hal_CoreConfig active = {
        .clock = HalClock_Hsi16_16MHz,
        .flags = HalCore_NoFlags,
    };
    if (config) active = *config;

    U32 sys_hz = hal__clock_init(active.clock);

    hal__core.clock = active.clock;
    hal__core.flags = active.flags;
    hal__core.sys_hz = sys_hz;
    hal__core.ahb_hz = sys_hz;
    hal__core.apb_hz = sys_hz;

    hal__systick_init(sys_hz);
}

Hal_Core const *hal_core(void) {
    return &hal__core;
}

U32 hal_now_us(void) {
    U32 primask = arch_irq_save();
    U32 ms = hal__core.tick_ms;
    U32 current = ARCH__SYSTICK_CVR;
    U32 csr = ARCH__SYSTICK_CSR;

    if ((csr & ARCH__SYSTICK_CSR_COUNTFLAG) != 0) {
        ms++;
        current = ARCH__SYSTICK_CVR;
    }

    arch_irq_restore(primask);

    U32 elapsed_cycles = hal__core.systick_cycles - current;
    U32 sub_us = elapsed_cycles >> 4;
    if (hal__core.sys_hz == HAL__PLL64_HZ) sub_us = elapsed_cycles >> 6;
    if (sub_us > 999u) sub_us = 999u;

    return (ms * 1000u) + sub_us;
}

void hal_watchdog_start(U32 timeout_ms) {
    HAL__RCC_APBENR1 |= HAL__RCC_APBENR1_DBGEN;
    arch_barrier();
    HAL__DBG_APB_FZ1 |= HAL__DBG_IWDG_STOP;

    HAL__IWDG_KR = 0xCCCCu;
    HAL__IWDG_KR = 0x5555u;
    HAL__IWDG_PR = 4u;                                   // LSI 32 kHz / 64 = 500 Hz
    HAL__IWDG_RLR = Min(timeout_ms / 2u, 4095u);
    for (U32 spin = 0; spin < 100000u && HAL__IWDG_SR != 0u; spin++) {
    }
    HAL__IWDG_KR = 0xAAAAu;
}

void hal_watchdog_feed(void) {
    HAL__IWDG_KR = 0xAAAAu;
}

void hal_delay_us(U32 us) {
    U32 start = hal_now_us();
    while (!hal_time_expired_us(start, us)) {
    }
}

bool hal_time_expired_us(U32 start_us, U32 duration_us) {
    return (U32)(hal_now_us() - start_us) >= duration_us;
}

void hal_system_reset(void) {
    arch_system_reset();
}

void hal_wait_for_interrupt(void) {
    arch_wait_for_interrupt();
}

bool hal_peripheral_prepare(Hal_PeripheralConfig const *config, Hal_PeripheralInfo *out) {
    if (!config) return false;
    Hal__PeripheralDesc const *desc = hal__peripheral_desc(config->peripheral);
    if (!desc) return false;

    U32 info_flags = HalPeripheralInfo_Prepared;

    if ((config->flags & HalPeripheralConfig_Reserve) != 0) {
        if (!hal__reserve(&hal__peripheral_reservations[config->peripheral], config->owner)) return false;
    }

    if ((config->flags & HalPeripheralConfig_Clock) != 0) {
        if (!hal__peripheral_clock_enable(config->peripheral)) return false;
        info_flags |= HalPeripheralInfo_Clocked;
    }

    if ((config->flags & HalPeripheralConfig_ResetPulse) != 0) {
        if (!hal__peripheral_reset_pulse(config->peripheral)) return false;
        info_flags |= HalPeripheralInfo_Reset;
    }

    if (out) {
        out->peripheral = config->peripheral;
        out->flags = info_flags;
        out->base = desc->base;
        out->bus_hz = hal__peripheral_bus_hz(config->peripheral);
        out->irq = desc->irq;
    }

    return true;
}

bool hal_pin_configure(Hal_PinConfig const *config) {
    if (!config) return false;
    if (!hal__valid_pin(config->pin)) return false;
    if (config->mode == HalPinMode_Alternate && config->alternate > 15u) return false;

    Hal_Peripheral gpio = hal__gpio_peripheral_for_port(config->pin.port);
    UPtr base = hal__gpio_base(config->pin.port);
    if (gpio == HalPeripheral_None || !base) return false;

    if ((config->flags & HalPinConfig_Reserve) != 0) {
        if (!hal__reserve(&hal__pin_reservations[config->pin.port][config->pin.index], config->owner)) return false;
    }

    if (!hal__peripheral_clock_enable(gpio)) return false;

    U32 pin = config->pin.index;
    U32 mode_bits = hal__pin_mode_bits(config->mode);
    U32 pull_bits = hal__pin_pull_bits(config->pull);
    U32 speed_bits = hal__pin_speed_bits(config->speed);

    if (config->mode == HalPinMode_Output) {
        if (config->initial_level == HalLevel_High) {
            HAL__GPIO_BSRR(base) = 1u << pin;
        } else if (config->initial_level == HalLevel_Low) {
            HAL__GPIO_BSRR(base) = 1u << (pin + 16u);
        }
    }

    if (config->output == HalOutput_OpenDrain) {
        HAL__GPIO_OTYPER(base) |= 1u << pin;
    } else {
        HAL__GPIO_OTYPER(base) &= ~(1u << pin);
    }

    HAL__PIN_FIELD_WRITE(HAL__GPIO_OSPEEDR(base), pin, 2u, speed_bits);
    HAL__PIN_FIELD_WRITE(HAL__GPIO_PUPDR(base), pin, 2u, pull_bits);

    if (config->mode == HalPinMode_Alternate) {
        if (pin < 8u) {
            HAL__PIN_FIELD_WRITE(HAL__GPIO_AFRL(base), pin, 4u, config->alternate);
        } else {
            HAL__PIN_FIELD_WRITE(HAL__GPIO_AFRH(base), pin & 7u, 4u, config->alternate);
        }
    }

    HAL__PIN_FIELD_WRITE(HAL__GPIO_MODER(base), pin, 2u, mode_bits);
    arch_barrier();
    return true;
}

bool hal_ucpd_dead_battery_release(Hal_UcpdDeadBatteryConfig const *config) {
    if (!config) return false;
    if (config->peripheral != HalPeripheral_Ucpd1 && config->peripheral != HalPeripheral_Ucpd2) return false;

    if ((config->flags & HalUcpdDeadBattery_Reserve) != 0) {
        if (!hal__reserve(&hal__ucpd_dead_battery_reservations[config->peripheral], config->owner)) return false;
    }

    if (!hal__peripheral_clock_enable(HalPeripheral_Syscfg)) return false;

    if (config->peripheral == HalPeripheral_Ucpd1) {
        HAL__SYSCFG_CFGR1 |= HAL__SYSCFG_CFGR1_UCPD1_STROBE;
    } else {
        HAL__SYSCFG_CFGR1 |= HAL__SYSCFG_CFGR1_UCPD2_STROBE;
    }
    arch_barrier();
    return true;
}

bool hal_usb_clock_prepare(Hal_UsbClockConfig const *config, Hal_UsbClockInfo *out) {
    if (!config) return false;
    if (config->source != HalUsbClock_Hsi48CrsUsbSof) return false;

    if ((config->flags & HalUsbClock_Reserve) != 0) {
        if (!hal__reserve(&hal__usb_clock_reservation, config->owner)) return false;
    }

    U32 info_flags = HalUsbClockInfo_Prepared;

    if ((config->flags & HalUsbClock_EnableVddio2) != 0) {
        if (!hal__peripheral_clock_enable(HalPeripheral_Pwr)) return false;
        HAL__PWR_CR2 = (HAL__PWR_CR2 & ~HAL__PWR_CR2_PVM_VDDIO2_MASK) |
                       HAL__PWR_CR2_PVM_VDDIO2_BYPASS;
        info_flags |= HalUsbClockInfo_Vddio2Enabled;
    }

    HAL__RCC_CR |= HAL__RCC_CR_HSI48ON;
    while ((HAL__RCC_CR & HAL__RCC_CR_HSI48RDY) == 0) {
    }
    info_flags |= HalUsbClockInfo_Hsi48Ready;

    HAL__RCC_CCIPR = (HAL__RCC_CCIPR & ~HAL__RCC_CCIPR_USBSEL_MASK) |
                     HAL__RCC_CCIPR_USBSEL_HSI48;

    if ((config->flags & HalUsbClock_EnableCrs) != 0) {
        if (!hal__peripheral_clock_enable(HalPeripheral_Crs)) return false;
        if (!hal__peripheral_reset_pulse(HalPeripheral_Crs)) return false;

        HAL__CRS_CR &= ~HAL__CRS_CR_CEN;
        HAL__CRS_ICR = HAL__CRS_ICR_ALL_FLAGS;
        HAL__CRS_CFGR = HAL__CRS_CFGR_SYNCSRC_USB_SOF |
                        HAL__CRS_CFGR_FELIM(34u) |
                        HAL__CRS_CFGR_RELOAD((HAL__HSI48_HZ / HAL__SYSTICK_HZ) - 1u);
        HAL__CRS_CR = (HAL__CRS_CR & HAL__CRS_CR_TRIM_MASK) |
                      HAL__CRS_CR_AUTOTRIMEN |
                      HAL__CRS_CR_CEN;
        info_flags |= HalUsbClockInfo_CrsConfigured;
    }

    arch_barrier();

    if (out) {
        out->flags = info_flags;
        out->usb_hz = HAL__HSI48_HZ;
        out->hsi48_hz = HAL__HSI48_HZ;
        out->crs_base = HAL__CRS_BASE;
        out->rcc_base = HAL__RCC_BASE;
    }

    return true;
}

void SysTick_Handler(void) {
    hal__core.tick_ms++;
    (void)ARCH__SYSTICK_CSR;
}

#endif

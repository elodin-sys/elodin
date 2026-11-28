// board.h — Aleph Flight Controller (rev 6) STM32H747 pin map used by the self-test.
// Source: modules/MCU/MCU.zen, modules/Sensors/Sensors.zen, spec.md.
// Alternate functions: ST STM32_open_pin_data, GPIO-STM32H747_gpio_v1_0.
#ifndef BOARD_H
#define BOARD_H

#include "hw.h"

// Status LEDs, active high (D19 red, D20 yellow-green, D21 blue).
static const Pin LED_RED = { GPIOE, 3 };
static const Pin LED_GREEN = { GPIOA, 15 };
static const Pin LED_BLUE = { GPIOI, 0 };

// USART3 to the debug probe's first console (AF7).
static const Pin CONSOLE_TX = { GPIOD, 8 };
static const Pin CONSOLE_RX = { GPIOD, 9 };
#define CONSOLE_AF 7u

// UART4 to the ExpressLRS ESP32-C3 UART0 (AF8); BOOT = GPIO9 strap, EN = CHIP_EN.
static const Pin ELRS_TX = { GPIOD, 1 };
static const Pin ELRS_RX = { GPIOI, 9 };
#define ELRS_AF 8u
static const Pin ELRS_BOOT = { GPIOG, 5 };
static const Pin ELRS_EN = { GPIOG, 6 };

// I2C buses, bit-banged. Pull-ups: I2C1/I2C4 to P3V3, I2C2/I2C3 to P3V3_SENS1.
static const I2cBus I2C_EEPROM = { "I2C1", { GPIOB, 8 }, { GPIOB, 9 } };
static const I2cBus I2C_IMU = { "I2C2", { GPIOF, 1 }, { GPIOF, 0 } };
static const I2cBus I2C_ENV = { "I2C3", { GPIOH, 7 }, { GPIOH, 8 } };
static const I2cBus I2C_GPS = { "I2C4", { GPIOH, 11 }, { GPIOH, 12 } };

// SPI buses, bit-banged (mode 0 unless noted).
static const SpiBus SPI_FRAM = { { GPIOE, 2 }, { GPIOE, 6 }, { GPIOE, 5 }, { GPIOE, 4 } };
static const SpiBus SPI_IMU = { { GPIOK, 0 }, { GPIOJ, 10 }, { GPIOJ, 11 }, { GPIOK, 1 } };
// SPI2 to the IMX-6 (U26), bit-banged in mode 3 by the INS test.
static const SpiBus SPI_INS = { { GPIOB, 10 }, { GPIOC, 3 }, { GPIOC, 2 }, { GPIOH, 3 } };

// Switched rails (high = on, pulled up on the board) and other control lines.
static const Pin SENS1_EN = { GPIOA, 4 };     // IMX-6 and I2C2/I2C3 sensors
static const Pin SENS2_EN = { GPIOJ, 5 };     // SPI5 BMI270
static const Pin SD_EN = { GPIOB, 7 };        // microSD
static const Pin P5V_AUX_EN = { GPIOH, 15 };  // TPS25210 eFuse for the JST connectors
static const Pin MCU_INT = { GPIOC, 0 };      // carrier interrupt, R1 pull-up
static const Pin IMU_INT = { GPIOG, 7 };      // SPI5 BMI270 INT1
static const Pin INS_DRDY = { GPIOA, 6 };     // IMX-6 G9: SPI strap at boot, then data ready
static const Pin INS_CS = { GPIOH, 3 };       // IMX-6 SPI2 CS, 10 kOhm pull-up

// USART6 to the debug probe's second console (AF7, ST open pin data).
static const Pin UART6_TX = { GPIOC, 6 };
static const Pin UART6_RX = { GPIOC, 7 };
#define UART6_AF 7u

// CAN transceivers (TJA1051, S tied low, VCC = P5V): TXD has an internal
// pull-up, RXD is a push-pull output.
static const Pin CAN1_TX = { GPIOH, 13 };  // U13, J10
static const Pin CAN1_RX = { GPIOH, 14 };
static const Pin CAN2_TX = { GPIOB, 6 };   // U12, J2 (shared with the carrier's U11)
static const Pin CAN2_RX = { GPIOB, 5 };

// Carrier SPI1/UART1 lines that a TXU0204 (U8, U10) drives toward the STM32.
static const Pin CARRIER_IN[] = { { GPIOG, 11 }, { GPIOD, 7 }, { GPIOA, 10 }, { GPIOG, 10 } };
static const char *const CARRIER_IN_NAME[] = { "SPI1 SCK PG11", "SPI1 MOSI PD7", "UART1 RX PA10", "SPI1 CS PG10" };

// Pins that leave the board on JST GH connectors or the microSD socket, with
// their board pulls (Connectors.zen, MCU.zen). Nothing else loads them.
enum { ExtNone, ExtUp, ExtDown };
typedef struct {
    const char *name;
    Pin pin;
    uint8_t ext;
} ExtPin;
static const ExtPin EXT_PINS[] = {
    { "J14 TX PD5", { GPIOD, 5 }, ExtNone },     { "J14 RX PD6", { GPIOD, 6 }, ExtNone },
    { "J14 CTS PD3", { GPIOD, 3 }, ExtDown },    { "J14 RTS PD4", { GPIOD, 4 }, ExtNone },
    { "J4 TX PF7", { GPIOF, 7 }, ExtNone },      { "J4 RX PF6", { GPIOF, 6 }, ExtNone },
    { "J5 TX PE1", { GPIOE, 1 }, ExtUp },        { "J5 RX PE0", { GPIOE, 0 }, ExtNone },
    { "J7 SCL PH11", { GPIOH, 11 }, ExtUp },     { "J7 SDA PH12", { GPIOH, 12 }, ExtUp },
    { "J8 1 PG3", { GPIOG, 3 }, ExtNone },       { "J8 2 PG2", { GPIOG, 2 }, ExtNone },
    { "J8 3 PK2", { GPIOK, 2 }, ExtNone },       { "J8 4 PG4", { GPIOG, 4 }, ExtNone },
    { "J6 TIM1_CH1 PE9", { GPIOE, 9 }, ExtNone },   { "J6 TIM1_CH2 PE11", { GPIOE, 11 }, ExtNone },
    { "J6 TIM1_CH3 PE13", { GPIOE, 13 }, ExtNone }, { "J6 TIM1_CH4 PE14", { GPIOE, 14 }, ExtNone },
    { "J6 TIM4_CH1 PD12", { GPIOD, 12 }, ExtNone }, { "J6 TIM4_CH2 PD13", { GPIOD, 13 }, ExtNone },
    { "J6 TIM4_CH3 PD14", { GPIOD, 14 }, ExtNone }, { "J6 TIM4_CH4 PD15", { GPIOD, 15 }, ExtNone },
    { "J9 TIM2_CH1 PA5", { GPIOA, 5 }, ExtNone },   { "J9 TIM2_CH2 PB3", { GPIOB, 3 }, ExtNone },
    { "J9 TIM2_CH3 PA2", { GPIOA, 2 }, ExtNone },   { "J9 TIM2_CH4 PA3", { GPIOA, 3 }, ExtNone },
    { "J9 TIM8_CH1 PI5", { GPIOI, 5 }, ExtNone },   { "J9 TIM8_CH2 PI6", { GPIOI, 6 }, ExtNone },
    { "J9 TIM8_CH3 PI7", { GPIOI, 7 }, ExtNone },   { "J9 TIM8_CH4 PI2", { GPIOI, 2 }, ExtNone },
    { "SD D0 PC8", { GPIOC, 8 }, ExtNone },      { "SD D1 PC9", { GPIOC, 9 }, ExtNone },
    { "SD D2 PC10", { GPIOC, 10 }, ExtNone },    { "SD D3 PC11", { GPIOC, 11 }, ExtNone },
    { "SD CLK PC12", { GPIOC, 12 }, ExtNone },   { "SD CMD PD2", { GPIOD, 2 }, ExtNone },
};
#define EXT_PIN_COUNT (sizeof EXT_PINS / sizeof EXT_PINS[0])

// Analog inputs (dedicated _C pads, DS12930 Rev 3 pin table).
#define ADC_CH_VBAT 0u   // PA0_C = ADC12_INP0, 147k/10k from VBAT_IN
#define ADC_CH_VIN 0u    // PC2_C = ADC3_INP0, 147k/10k from VIN
#define ADC_CH_IMON 1u   // PC3_C = ADC3_INP1, TPS25210 IMON, 0.200 V/A
#define ADC_CH_VREFINT 19u  // ADC3 internal (ADC_VER_V5_X)
#define ADC_CH_VBAT_INT 17u // ADC3 internal: RTC backup cell (BT1) through a 1/4 ladder
#define ADC_CH_TEMP 18u     // ADC3 internal temperature sensor
// stm32h7xx_ll_adc.h: TS_CAL1 at 30 C, TS_CAL2 at 130 C (H747), both at 3.3 V.
#define TS_CAL1_ADDR 0x1FF1E820u
#define TS_CAL2_ADDR 0x1FF1E840u
#define DIVIDER_NUM 157u    // (147k + 10k) / 10k, times 10
#define DIVIDER_DEN 10u

// Device addresses and identities (datasheets in README).
#define EEPROM_ADDR 0x50u   // CAT24C512, A2..A0 low
#define BMI270_ADDR 0x68u   // SDO low
#define BMI270_CHIP_ID 0x24u
#define BMM350_ADDR 0x14u   // ADSEL low
#define BMM350_CHIP_ID 0x33u
#define BMP581_ADDR 0x46u   // SDO low
#define BMP581_CHIP_ID 0x50u
#define BMP384_ADDR 0x76u   // BOM alternate
#define BMP384_CHIP_ID 0x50u

#endif

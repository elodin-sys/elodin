// selftest.c — Aleph Flight Controller bring-up self-test for the STM32H747 (M7 core).
//
// Runs once at reset and again on 'r' from the console, reporting one line per
// check on USART3, which the debug probe exposes as "Aleph FC UART3"
// (115200 8N1). The LEDs show progress: blue while running, then green if
// nothing failed or red if something did.
//
// Everything stays on the 64 MHz HSI with the LDO supply. Sensors and memories
// are reached by bit-banged I2C/SPI, so a wiring or assembly fault shows up as
// a failed read rather than a misconfigured peripheral.

#include "board.h"
#include "hw.h"

uint32_t SystemCoreClock = CORE_HZ;
volatile uint32_t tick_ms;

void SysTick_Handler(void) {
    tick_ms++;
}

static uint32_t passes, fails, warnings;

static void report(const char *level, const char *name, const char *fmt, va_list ap) {
    con_printf("[%s] %s", level, name);
    if (fmt && *fmt) {
        con_puts(": ");
        con_vprintf(fmt, ap);
    }
    con_putc('\n');
}

static void pass(const char *name, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    report("PASS", name, fmt, ap);
    va_end(ap);
    passes++;
}

static void fail(const char *name, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    report("FAIL", name, fmt, ap);
    va_end(ap);
    fails++;
}

static void warn(const char *name, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    report("WARN", name, fmt, ap);
    va_end(ap);
    warnings++;
}

static void info(const char *name, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    report("INFO", name, fmt, ap);
    va_end(ap);
}

static void check(bool ok, const char *name, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    report(ok ? "PASS" : "FAIL", name, fmt, ap);
    va_end(ap);
    if (ok) passes++;
    else fails++;
}

static bool wait_flag(volatile uint32_t *reg, uint32_t mask, bool set, uint32_t timeout_ms) {
    uint32_t start = now_ms();
    while (((*reg & mask) != 0u) != set) {
        if ((uint32_t)(now_ms() - start) > timeout_ms) return false;
    }
    return true;
}

static void leds(bool red, bool green, bool blue) {
    pin_write(LED_RED, red);
    pin_write(LED_GREEN, green);
    pin_write(LED_BLUE, blue);
}

// ---- Identity and configuration ----

static void test_identity(void) {
    uint32_t idcode = DBGMCU->IDCODE;
    uint32_t dev = idcode & DBGMCU_IDCODE_DEV_ID;
    uint32_t rev = idcode >> 16;
    const char *rev_name = rev == 0x1003u ? "Y" : rev == 0x2003u ? "V" : "?";
    check(dev == 0x450u, "device id", "0x%03x (expect 0x450, STM32H745/747/755/757), rev 0x%04x (%s)",
          dev, rev, rev_name);

    uint32_t flash_kb = *(volatile uint16_t *)FLASHSIZE_BASE;
    check(flash_kb == 2048u, "flash size", "%u KB (expect 2048 for STM32H747XI)", flash_kb);

    volatile uint32_t *uid = (volatile uint32_t *)UID_BASE;
    info("unique id", "%08x %08x %08x", uid[2], uid[1], uid[0]);

    uint32_t optsr = FLASH->OPTSR_CUR;
    uint32_t rdp = (optsr & FLASH_OPTSR_RDP_Msk) >> FLASH_OPTSR_RDP_Pos;
    check(rdp == 0xAAu, "readout protection", "RDP 0x%02x (0xAA = level 0)", rdp);
    info("core boot", "BCM7=%u BCM4=%u, M7 boot 0x%04x0000, M4 boot 0x%04x0000",
         (optsr & FLASH_OPTSR_BCM7) ? 1u : 0u, (optsr & FLASH_OPTSR_BCM4) ? 1u : 0u,
         (FLASH->BOOT7_CUR & FLASH_BOOT7_BCM7_ADD0_Msk) >> FLASH_BOOT7_BCM7_ADD0_Pos,
         (FLASH->BOOT4_CUR & FLASH_BOOT4_BCM4_ADD0_Msk) >> FLASH_BOOT4_BCM4_ADD0_Pos);
}

static uint32_t reset_flags;

static void test_reset_cause(void) {
    uint32_t r = reset_flags;
    info("reset cause", "RSR 0x%08x%s%s%s%s%s", r,
         (r & RCC_RSR_PORRSTF) ? " POR" : "", (r & RCC_RSR_BORRSTF) ? " BOR" : "",
         (r & RCC_RSR_PINRSTF) ? " PIN" : "", (r & RCC_RSR_SFT1RSTF) ? " SW" : "",
         (r & (RCC_RSR_IWDG1RSTF | RCC_RSR_WWDG1RSTF)) ? " WDG" : "");
}

static void test_supply(void) {
    uint32_t cr3 = PWR->CR3;
    bool ldo = (cr3 & PWR_CR3_LDOEN) != 0u && (cr3 & PWR_CR3_SMPSEN) == 0u;
    check(ldo, "supply", "PWR_CR3 0x%08x (LDO on, SMPS off; VDDSMPS is grounded)", cr3);
    check((PWR->CSR1 & PWR_CSR1_ACTVOSRDY) != 0u, "core regulator", "ACTVOSRDY");
}

// ---- Clocks ----

static void test_clocks(void) {
    uint32_t start = now_ms();
    RCC->CR |= RCC_CR_HSEON;  // crystal, not bypass
    bool hse = wait_flag(&RCC->CR, RCC_CR_HSERDY, true, 100u);
    check(hse, "HSE 24 MHz", hse ? "ready after %u ms" : "not ready after 100 ms", now_ms() - start);

    if (hse) {
        // PLL1 from HSE: 24 MHz / 4 = 6 MHz input (4-8 MHz range), x80 = 480 MHz
        // VCO (wide range). Lock only; the system clock stays on HSI.
        RCC->PLLCKSELR = (RCC->PLLCKSELR & ~(RCC_PLLCKSELR_PLLSRC | RCC_PLLCKSELR_DIVM1)) |
                         RCC_PLLCKSELR_PLLSRC_HSE | (4u << RCC_PLLCKSELR_DIVM1_Pos);
        RCC->PLLCFGR = (RCC->PLLCFGR & ~(RCC_PLLCFGR_PLL1RGE | RCC_PLLCFGR_PLL1VCOSEL | RCC_PLLCFGR_PLL1FRACEN)) |
                       RCC_PLLCFGR_PLL1RGE_2;
        RCC->PLL1DIVR = ((80u - 1u) << RCC_PLL1DIVR_N1_Pos) | ((2u - 1u) << RCC_PLL1DIVR_P1_Pos) |
                        ((4u - 1u) << RCC_PLL1DIVR_Q1_Pos) | ((2u - 1u) << RCC_PLL1DIVR_R1_Pos);
        RCC->CR |= RCC_CR_PLL1ON;
        bool lock = wait_flag(&RCC->CR, RCC_CR_PLL1RDY, true, 10u);
        check(lock, "PLL1 from HSE", lock ? "locked at 480 MHz VCO" : "no lock in 10 ms");
        RCC->CR &= ~RCC_CR_PLL1ON;
    }

    PWR->CR1 |= PWR_CR1_DBP;
    wait_flag(&PWR->CR1, PWR_CR1_DBP, true, 10u);
    start = now_ms();
    if ((RCC->BDCR & RCC_BDCR_LSERDY) == 0u) {
        RCC->BDCR = (RCC->BDCR & ~RCC_BDCR_LSEDRV) | (3u << RCC_BDCR_LSEDRV_Pos) | RCC_BDCR_LSEON;
    }
    bool lse = wait_flag(&RCC->BDCR, RCC_BDCR_LSERDY, true, 3000u);
    check(lse, "LSE 32.768 kHz", lse ? "ready after %u ms (high drive)" : "not ready after 3 s",
          now_ms() - start);
}

// ---- Static levels on pulled lines ----

static void expect_high(Pin p, const char *name, const char *why) {
    pin_input(p, PullNone);
    delay_us(50);
    check(pin_read(p), name, "%s (%s)", pin_read(p) ? "high" : "LOW", why);
}

static void test_static_levels(void) {
    expect_high(SENS1_EN, "SENS1_EN PA4", "R60 pull-up, rail on");
    expect_high(SENS2_EN, "SENS2_EN PJ5", "R70 pull-up, rail on");
    expect_high(SD_EN, "SD_EN PB7", "R54 pull-up, rail on");
    expect_high(P5V_AUX_EN, "P5V_AUX_EN PH15", "R21 pull-up, eFuse on");
    expect_high(ELRS_BOOT, "ELRS BOOT PG5", "receiver pull-up, SPI boot");
    expect_high(ELRS_EN, "ELRS EN PG6", "receiver pull-up, running");
    expect_high(INS_CS, "INS CS PH3", "10 kOhm pull-up");
    expect_high(SPI_FRAM.cs, "F-RAM CS PE4", "R95 pull-up");
    pin_input(MCU_INT, PullNone);
    delay_us(50);
    if (pin_read(MCU_INT)) pass("MCU_INT PC0", "high (R1 pull-up)");
    else warn("MCU_INT PC0", "low (driven by the carrier, or R1 missing)");
}

// ---- I2C devices ----

static bool bus_up(const I2cBus *b) {
    bool ok = i2c_init(b);
    check(ok, b->name, ok ? "SCL and SDA idle high" : "SCL %s, SDA %s: missing pull-up, rail off or stuck",
          pin_read(b->scl) ? "high" : "LOW", pin_read(b->sda) ? "high" : "LOW");
    return ok;
}

static void expect_chip_id(const I2cBus *b, uint8_t addr, uint8_t reg, uint8_t id, int dummy, const char *name) {
    uint8_t v = 0;
    bool acked = i2c_read_reg(b, addr, reg, &v, 1, dummy);
    if (!acked) fail(name, "no ACK at 0x%02x on %s", addr, b->name);
    else check(v == id, name, "chip id 0x%02x at 0x%02x (expect 0x%02x)", v, addr, id);
}

// Cycles at 64 MHz from releasing `p` (held low 20 us) until it reads high,
// or UINT32_MAX after 1 ms.
static uint32_t rise_cycles(Pin p) {
    i2c_pull(p);
    delay_us(20);
    uint32_t start = DWT->CYCCNT;
    i2c_release(p);
    while (!pin_read(p)) {
        if (DWT->CYCCNT - start > CORE_HZ / 1000u) return UINT32_MAX;
    }
    return DWT->CYCCNT - start;
}

static void scan(const I2cBus *b) {
    con_printf("[INFO] %s scan:", b->name);
    int found = 0;
    for (uint8_t a = 0x08; a < 0x78; a++) {
        if (i2c_probe(b, a)) {
            con_printf(" 0x%02x", a);
            found++;
        }
    }
    con_puts(found ? "\n" : " nothing answered\n");
}

// Line health for a bus whose devices don't answer: rise times against the
// board pull-ups, and whether pulling one line drags the other.
static void line_diag(const I2cBus *b) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t scl = rise_cycles(b->scl), sda = rise_cycles(b->sda);
    con_printf("[INFO] %s rise to high after release: SCL %u ns, SDA %u ns\n", b->name,
               scl == UINT32_MAX ? 0u : scl * 1000u / 64u, sda == UINT32_MAX ? 0u : sda * 1000u / 64u);
    i2c_pull(b->scl);
    delay_us(20);
    bool sda_with_scl_low = pin_read(b->sda);
    i2c_release(b->scl);
    i2c_pull(b->sda);
    delay_us(20);
    bool scl_with_sda_low = pin_read(b->scl);
    i2c_release(b->sda);
    delay_us(20);
    con_printf("[INFO] %s crosstalk: SCL low -> SDA %s, SDA low -> SCL %s\n", b->name,
               sda_with_scl_low ? "high" : "LOW", scl_with_sda_low ? "high" : "LOW");
    // Address both sensors bit by bit, reading SDA and SCL back while SCL is
    // high: a part dragging either line shows up as a mismatch or a stretch.
    static const uint8_t addrs[] = { 0x14, 0x46 };
    for (unsigned i = 0; i < sizeof addrs; i++) {
        uint8_t byte = (uint8_t)(addrs[i] << 1), seen = 0;
        bool stretched = false;
        i2c_start(b);
        for (int bit = 7; bit >= 0; bit--) {
            pin_write(b->sda, (byte >> bit) & 1u);
            delay_us(I2C_HALF_US);
            i2c_release(b->scl);
            delay_us(2);
            if (!pin_read(b->scl)) stretched = true;
            seen = (uint8_t)((seen << 1) | (pin_read(b->sda) ? 1u : 0u));
            delay_us(I2C_HALF_US);
            i2c_pull(b->scl);
        }
        i2c_release(b->sda);
        delay_us(I2C_HALF_US);
        i2c_release(b->scl);
        delay_us(2);
        bool ack = !pin_read(b->sda);
        delay_us(I2C_HALF_US);
        i2c_pull(b->scl);
        i2c_stop(b);
        con_printf("[INFO] %s address 0x%02x: sent %02x, SDA read back %02x, SCL %s, %s\n", b->name, addrs[i], byte,
                   seen, stretched ? "held LOW" : "free", ack ? "ACK" : "no ACK");
    }
    scan(b);
}

// P3V3_SENS1 has no ADC channel, but the I2C2 SCL pull-up (R12) returns to it.
// Stops the SPI2 peripheral if the IMX-6 test left it running (suspend,
// then disable, as RM0399 asks for an endless master transfer).
static void ins_hw_stop(void) {
    if ((RCC->APB1LENR & RCC_APB1LENR_SPI2EN) == 0u || (SPI2->CR1 & SPI_CR1_SPE) == 0u) return;
    SPI2->CR1 |= SPI_CR1_CSUSP;
    for (uint32_t n = 0; (SPI2->SR & SPI_SR_SUSP) == 0u && n < 100000u; n++) {
    }
    SPI2->IFCR = SPI_IFCR_SUSPC;
    SPI2->CR1 &= ~SPI_CR1_SPE;
}

// Floats the IMX-6 SPI2 lines, so nothing back-powers the module while
// P3V3_SENS1 is off (CS falls with the rail through its pull-up).
static void ins_release(void) {
    ins_hw_stop();
    pin_input(SPI_INS.sck, PullNone);
    pin_input(SPI_INS.mosi, PullNone);
    pin_input(SPI_INS.miso, PullNone);
    pin_input(SPI_INS.cs, PullNone);
}

// Cycle the rail with PA4 and watch that line for 50 ms: a rail that rises and
// falls again is hitting U29's current limit, one that never rises is off.
static void sens1_diag(void) {
    const Pin probe_line = I2C_IMU.scl;
    ins_release();
    pin_input(I2C_IMU.scl, PullNone);
    pin_input(I2C_IMU.sda, PullNone);
    pin_input(I2C_ENV.scl, PullNone);
    pin_input(I2C_ENV.sda, PullNone);
    pin_output(SENS1_EN, false, false);
    delay_ms(200);
    bool off_level = pin_read(probe_line);
    pin_write(SENS1_EN, true);
    uint32_t start = now_us(), first_high = 0, last_high = 0, highs = 0, samples = 0, drops = 0;
    bool prev = false;
    while ((uint32_t)(now_us() - start) < 50000u) {
        bool h = pin_read(probe_line);
        uint32_t t = now_us() - start;
        if (h) {
            if (!highs) first_high = t;
            last_high = t;
            highs++;
        } else if (prev) {
            drops++;
        }
        prev = h;
        samples++;
        delay_us(10);
    }
    pin_input(SENS1_EN, PullNone);
    if (!highs) {
        fail("P3V3_SENS1 rail", "I2C2 SCL (R12 to the rail) never rose in 50 ms after PA4 went high (rail off: U29, its EN or a hard short)");
    } else if (prev) {
        check(drops == 0, "P3V3_SENS1 rail", "I2C2 SCL high %u us after enable, %u of %u samples high, %u drops", first_high,
              highs, samples, drops);
    } else {
        fail("P3V3_SENS1 rail", "rose %u us after enable, last high at %u us, %u drops: the rail collapses (overload or short)",
             first_high, last_high, drops);
    }
    (void)off_level;
}

static void test_i2c(void) {
    if (bus_up(&I2C_EEPROM)) {
        check(i2c_probe(&I2C_EEPROM, EEPROM_ADDR), "board-ID EEPROM", "CAT24C512 ACK at 0x%02x", EEPROM_ADDR);
    }
    if (bus_up(&I2C_IMU)) {
        expect_chip_id(&I2C_IMU, BMI270_ADDR, 0x00, BMI270_CHIP_ID, 0, "BMI270 (I2C2, U16)");
    } else {
        sens1_diag();
    }
    if (bus_up(&I2C_ENV)) {
        // BMM350 returns two dummy bytes before data on every I2C read.
        expect_chip_id(&I2C_ENV, BMM350_ADDR, 0x00, BMM350_CHIP_ID, 2, "BMM350 (I2C3, U19)");
        if (i2c_probe(&I2C_ENV, BMP581_ADDR)) {
            expect_chip_id(&I2C_ENV, BMP581_ADDR, 0x01, BMP581_CHIP_ID, 0, "BMP581 (I2C3, U15)");
        } else if (i2c_probe(&I2C_ENV, BMP384_ADDR)) {
            expect_chip_id(&I2C_ENV, BMP384_ADDR, 0x00, BMP384_CHIP_ID, 0, "BMP384 (I2C3, U15 alternate)");
        } else {
            fail("barometer (I2C3, U15)", "no ACK at 0x%02x (BMP581) or 0x%02x (BMP384)", BMP581_ADDR, BMP384_ADDR);
        }
    }
    line_diag(&I2C_ENV);
    if (i2c_init(&I2C_GPS)) {
        con_puts("[INFO] I2C4 (J7) scan:");
        int found = 0;
        for (uint8_t a = 0x08; a < 0x78; a++) {
            if (i2c_probe(&I2C_GPS, a)) {
                con_printf(" 0x%02x", a);
                found++;
            }
        }
        con_puts(found ? "\n" : " nothing attached\n");
    } else {
        warn("I2C4 (J7)", "lines not idle high: check pull-ups R16/R17");
    }
}

// ---- SPI devices ----

static void fram_cmd(uint8_t op) {
    spi_select(&SPI_FRAM, true);
    spi_byte(&SPI_FRAM, op);
    spi_select(&SPI_FRAM, false);
}

static uint8_t fram_read(uint16_t addr) {
    spi_select(&SPI_FRAM, true);
    spi_byte(&SPI_FRAM, 0x03);  // READ
    spi_byte(&SPI_FRAM, (uint8_t)(addr >> 8));
    spi_byte(&SPI_FRAM, (uint8_t)addr);
    uint8_t v = spi_byte(&SPI_FRAM, 0);
    spi_select(&SPI_FRAM, false);
    return v;
}

static void fram_write(uint16_t addr, uint8_t v) {
    fram_cmd(0x06);  // WREN
    spi_select(&SPI_FRAM, true);
    spi_byte(&SPI_FRAM, 0x02);  // WRITE
    spi_byte(&SPI_FRAM, (uint8_t)(addr >> 8));
    spi_byte(&SPI_FRAM, (uint8_t)addr);
    spi_byte(&SPI_FRAM, v);
    spi_select(&SPI_FRAM, false);
}

static void test_fram(void) {
    spi_init(&SPI_FRAM);
    uint8_t id[9];
    spi_select(&SPI_FRAM, true);
    spi_byte(&SPI_FRAM, 0x9F);  // RDID
    for (int i = 0; i < 9; i++) id[i] = spi_byte(&SPI_FRAM, 0);
    spi_select(&SPI_FRAM, false);

    // FM25V02A (Cypress 001-90865): 7F x6, C2, then family/density 0x22 and
    // sub/rev/reserved 0b00RRR000 (rev 1 reads 0x08).
    bool ok = id[6] == 0xC2u && id[7] == 0x22u && (id[8] & 0xC7u) == 0u;
    for (int i = 0; i < 6; i++) ok = ok && id[i] == 0x7Fu;
    check(ok, "F-RAM FM25V02A (SPI4)", "RDID %02x%02x%02x%02x%02x%02x%02x%02x%02x, rev %u", id[0], id[1], id[2],
          id[3], id[4], id[5], id[6], id[7], id[8], (id[8] >> 3) & 7u);
    if (!ok) return;

    // Write and restore the last byte (parameters live at the low addresses).
    const uint16_t addr = 0x7FFF;
    uint8_t saved = fram_read(addr);
    uint8_t probe = (uint8_t)~saved;
    fram_write(addr, probe);
    uint8_t got = fram_read(addr);
    fram_write(addr, saved);
    bool restored = fram_read(addr) == saved;
    check(got == probe && restored, "F-RAM write/read", "0x%04x: wrote 0x%02x, read 0x%02x, %s", addr, probe, got,
          restored ? "restored" : "RESTORE FAILED");
}

static uint8_t bmi_read(uint8_t reg) {
    spi_select(&SPI_IMU, true);
    spi_byte(&SPI_IMU, (uint8_t)(0x80u | reg));
    spi_byte(&SPI_IMU, 0);  // one dummy byte on SPI reads
    uint8_t v = spi_byte(&SPI_IMU, 0);
    spi_select(&SPI_IMU, false);
    delay_us(1000);  // > 450 us between accesses in advanced power save (the reset default)
    return v;
}

static void bmi_write(uint8_t reg, uint8_t v) {
    spi_select(&SPI_IMU, true);
    spi_byte(&SPI_IMU, reg & 0x7Fu);
    spi_byte(&SPI_IMU, v);
    spi_select(&SPI_IMU, false);
    delay_us(1000);  // > 450 us between accesses in advanced power save
}

static bool bmi_spi_up(void) {
    spi_init(&SPI_IMU);
    delay_us(1000);
    (void)bmi_read(0x00);  // the interface starts in I2C; a dummy read selects SPI
    return bmi_read(0x00) == BMI270_CHIP_ID;
}

static void test_spi_imu(void) {
    bool up = bmi_spi_up();
    check(up, "BMI270 (SPI5, U18)", "chip id 0x%02x (expect 0x%02x)", bmi_read(0x00), BMI270_CHIP_ID);
    if (!up) return;

    // Prove PJ5 really switches P3V3_SENS2: set ACC_RANGE (0x41, reset 0x02) to
    // 0x03, cycle the rail, and expect the reset value back. While the rail is
    // off, CS, SCK and MOSI are held low so nothing back-powers the sensor.
    bmi_write(0x41, 0x03);
    uint8_t before = bmi_read(0x41);
    pin_write(SPI_IMU.cs, false);
    pin_output(SENS2_EN, false, false);
    delay_ms(100);
    pin_write(SENS2_EN, true);
    pin_input(SENS2_EN, PullNone);  // back to the R70 pull-up
    delay_ms(10);
    bool back = bmi_spi_up();
    uint8_t after = bmi_read(0x41);
    check(before == 0x03u && back && after == 0x02u, "SENS2 rail (PJ5)",
          "ACC_RANGE 0x%02x before, 0x%02x after a 100 ms off cycle (expect 0x03, 0x02)", before, after);

    pin_input(IMU_INT, PullNone);
    info("BMI270 INT1 (PG7)", "%s at idle (high impedance until configured)", pin_read(IMU_INT) ? "high" : "low");
}

// ---- IMX-6 INS (SPI2, Inertial Sense binary protocol) ----
//
// Inertial Sense SPI spec (docs.inertialsense.com, Communication Protocols >
// SPI): mode 3, CS held low through a transfer, at least one bit time between
// bytes, zeros while the module has nothing to send, DR (G9) high while data
// waits, up to 3 Mbit/s polled or 5 Mbit/s DR-gated. Packets (SDK
// src/ISComm.h, protocol 2): EF 49, flags (packet type in the low nibble,
// 0x20 = payload starts with a 2-byte offset), data id, payload size (LE16),
// payload, then a Fletcher-16 of everything before it, low byte first.
// Structure offsets are from SDK src/data_sets.h (protocol 2.2.x.1).

#define ISB_ACK 1u
#define ISB_NACK 2u
#define ISB_GET_DATA 3u
#define ISB_DATA 4u
#define ISB_SET_DATA 5u
#define ISB_STOP_DID 7u          // stops one data set's broadcast, id in the header, no reply
#define ISB_FLAG_OFFSET 0x20u
#define DID_DEV_INFO 1u          // dev_info_t, 84 bytes
#define DID_SYS_FAULT 2u         // system_fault_t: upTime, status, fileNum, lineNum, haltReason, lr, pc, psr, ...
#define DID_PIMU 3u              // pimu_t, 40 bytes
#define DID_INS_1 4u             // ins_1_t, 80 bytes
#define DID_SYS_CMD 7u           // system_command_t: command, ~command
#define DID_SYS_PARAMS 10u       // sys_params_t, 64 bytes
#define DID_FLASH_CONFIG 12u     // nvm_flash_cfg_t, 288 bytes
#define DID_BIT 64u              // bit_t, 56 bytes: command 0, state 2, hdwBitStatus 4, calBitStatus 8, pqr 36, acc 40
#define DID_MAGNETOMETER 52u     // magnetometer_t: time (double), mag[3] uT
#define DID_BAROMETER 53u        // barometer_t: time, bar kPa, mslBar, barTemp C, humidity
#define FLASH_CFG_CHECKSUM 4u    // XOR of the 32-bit words after size and checksum
#define FLASH_CFG_IOCONFIG 132u
#define IO_CONFIG_G9_MASK 0x30u
#define IO_CONFIG_G9_SPI_DRDY 0x30u
#define SYS_CMD_SOFTWARE_RESET 99u
#define HDW_STATUS_SATURATION 0x00000F00u
#define HDW_STATUS_FLASH_WRITE_PENDING 0x00008000u
#define HDW_STATUS_BIT_MASK 0x03000000u
#define HDW_STATUS_BIT_PASSED 0x02000000u
#define HDW_STATUS_BIT_FAILED 0x03000000u
#define HDW_STATUS_SPI_ENABLED 0x08000000u
#define HDW_STATUS_RESET_CAUSE 0x70000000u
#define HDW_STATUS_CRITICAL 0x80000000u
#define IS_HDW_IMX 3u            // eIsHardwareType
#define IS_RUN_BOOTLOADER 1u     // eHdwRunStates
#define IS_RUN_APP 2u

static uint16_t le16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float lef(const uint8_t *p) {
    union {
        uint32_t u;
        float f;
    } v = { le32(p) };
    return v.f;
}

static double led(const uint8_t *p) {
    union {
        uint64_t u;
        double d;
    } v = { (uint64_t)le32(p) | ((uint64_t)le32(p + 4) << 32) };
    return v.d;
}

static void put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint16_t fletcher16(const uint8_t *p, uint32_t n) {
    uint8_t a = 0, b = 0;
    while (n--) {
        a = (uint8_t)(a + *p++);
        b = (uint8_t)(b + a);
    }
    return (uint16_t)(a | (b << 8));
}

// Float to int with NaN as 0 and a bound that keeps the squares in range.
static int32_t to_int(float v, float limit) {
    if (v != v) return 0;
    if (v > limit) v = limit;
    if (v < -limit) v = -limit;
    return (int32_t)v;
}

static float fsqrt(float x) {
    if (!(x > 0.0f)) return 0.0f;
    float r = x > 1.0f ? x : 1.0f;
    for (int i = 0; i < 40; i++) r = 0.5f * (r + x / r);
    return r;
}

typedef struct {
    uint8_t buf[160];  // header + payload + checksum of the packets we keep
    uint16_t n, need, skip;
    uint32_t bad;  // packets that failed the checksum
} IsbRx;

static IsbRx ins_rx;

// Takes one received byte; true once buf holds a complete packet with a good
// checksum. Packets too big for buf are skipped whole.
static bool isb_feed(IsbRx *r, uint8_t b) {
    if (r->skip) {
        r->skip--;
        return false;
    }
    if (r->n == 1 && b != 0x49u) r->n = 0;
    if (r->n == 0 && b != 0xEFu) return false;
    r->buf[r->n++] = b;
    if (r->n == 6u) {
        r->need = (uint16_t)(6u + le16(&r->buf[4]) + 2u);
        if (r->need > sizeof r->buf) {
            r->skip = (uint16_t)(r->need - 6u);
            r->n = 0;
            return false;
        }
    }
    if (r->n < 6u || r->n < r->need) return false;
    r->n = 0;
    if (le16(&r->buf[r->need - 2u]) != fletcher16(r->buf, r->need - 2u)) {
        r->bad++;
        return false;
    }
    return true;
}

static bool isb_idle(const IsbRx *r) {
    return r->n == 0u && r->skip == 0u;
}

// Builds a packet the way the SDK's is_comm_write does: the offset field (and
// its flag) only when the offset is not zero.
static uint32_t isb_packet(uint8_t *p, uint8_t type, uint8_t id, uint16_t offset, const uint8_t *data, uint16_t n) {
    uint32_t len = 0;
    bool off = offset != 0u;
    uint16_t size = (uint16_t)(n + (off ? 2u : 0u));
    p[len++] = 0xEF;
    p[len++] = 0x49;
    p[len++] = (uint8_t)(type | (off ? ISB_FLAG_OFFSET : 0u));
    p[len++] = id;
    p[len++] = (uint8_t)size;
    p[len++] = (uint8_t)(size >> 8);
    if (off) {
        p[len++] = (uint8_t)offset;
        p[len++] = (uint8_t)(offset >> 8);
    }
    for (uint16_t i = 0; i < n; i++) p[len++] = data[i];
    uint16_t ck = fletcher16(p, len);
    p[len++] = (uint8_t)ck;
    p[len++] = (uint8_t)(ck >> 8);
    return len;
}

// ---- Transport: bit-banged at about 200 kHz, or the SPI2 peripheral ----

static bool ins_hw;
static bool ins_dr_seen, ins_dr_last;
static uint32_t ins_dr_rises;

static void ins_sample_dr(void) {
    bool v = pin_read(INS_DRDY);
    if (v) ins_dr_seen = true;
    if (v && !ins_dr_last) ins_dr_rises++;
    ins_dr_last = v;
}

// Mode 3: SCK idles high, the module shifts on the falling edge and samples on
// the rising edge.
static uint8_t ins_bb_byte(uint8_t out) {
    uint8_t in = 0;
    for (int i = 7; i >= 0; i--) {
        pin_write(SPI_INS.sck, false);
        pin_write(SPI_INS.mosi, (out >> i) & 1u);
        delay_us(SPI_HALF_US);
        in = (uint8_t)((in << 1) | (pin_read(SPI_INS.miso) ? 1u : 0u));
        pin_write(SPI_INS.sck, true);
        delay_us(SPI_HALF_US);
    }
    delay_us(SPI_HALF_US);  // the spec asks for a gap of at least one bit time
    return in;
}

static uint32_t ins_hw_timeouts;

static uint8_t ins_hw_byte(uint8_t out) {
    uint32_t n = 0;
    while ((SPI2->SR & SPI_SR_TXP) == 0u) {
        if (++n > 100000u) {
            ins_hw_timeouts++;
            return 0;
        }
    }
    *(volatile uint8_t *)&SPI2->TXDR = out;
    n = 0;
    while ((SPI2->SR & SPI_SR_RXP) == 0u) {
        if (++n > 100000u) {
            ins_hw_timeouts++;
            return 0;
        }
    }
    return *(volatile uint8_t *)&SPI2->RXDR;
}

static uint8_t ins_byte(uint8_t out) {
    return ins_hw ? ins_hw_byte(out) : ins_bb_byte(out);
}

static void ins_select(bool on) {
    delay_us(2);
    pin_write(SPI_INS.cs, !on);
    delay_us(2);
}

static void ins_rx_reset(void) {
    ins_rx.n = ins_rx.skip = 0;
    ins_rx.bad = 0;
}

static void ins_open_bitbang(void) {
    ins_hw_stop();
    ins_hw = false;
    pin_input(INS_DRDY, PullNone);  // never drive: G9 is the module's SPI strap
    pin_output(SPI_INS.cs, true, false);
    pin_output(SPI_INS.sck, true, false);
    pin_output(SPI_INS.mosi, false, false);
    pin_input(SPI_INS.miso, PullDown);  // an absent module reads as idle zeros
    ins_rx_reset();
}

// SPI2 master, mode 3, 8-bit, one idle clock between bytes (MIDI), kernel
// clock per_ck = HSI 64 MHz, SCK = 64 MHz / 2^(mbr + 1).
static void ins_open_hw(uint32_t mbr) {
    ins_hw_stop();
    pin_input(INS_DRDY, PullNone);
    pin_output(SPI_INS.cs, true, false);
    RCC->D2CCIP1R = (RCC->D2CCIP1R & ~RCC_D2CCIP1R_SPI123SEL_Msk) | (4u << RCC_D2CCIP1R_SPI123SEL_Pos);
    RCC->APB1LENR |= RCC_APB1LENR_SPI2EN;
    (void)RCC->APB1LENR;
    // SSI high before MASTER/SSM, or the SPI latches a mode fault (MODF) and
    // refuses SPE until it is cleared.
    SPI2->CR1 = SPI_CR1_SSI;
    SPI2->IFCR = SPI_IFCR_MODFC;
    SPI2->CFG1 = (mbr << SPI_CFG1_MBR_Pos) | (7u << SPI_CFG1_DSIZE_Pos);
    SPI2->CFG2 = SPI_CFG2_AFCNTR | SPI_CFG2_SSM | SPI_CFG2_CPOL | SPI_CFG2_CPHA | SPI_CFG2_MASTER |
                 (1u << SPI_CFG2_MIDI_Pos);
    SPI2->CR2 = 0;  // TSIZE 0: endless transfer, one frame per TXDR write
    SPI2->IFCR = SPI_IFCR_MODFC;
    pin_config(SPI_INS.sck, ModeAf, false, PullNone, 5);  // AF5 = SPI2 on PB10, PC2, PC3
    pin_config(SPI_INS.mosi, ModeAf, false, PullNone, 5);
    pin_config(SPI_INS.miso, ModeAf, false, PullDown, 5);
    SPI_INS.sck.port->OSPEEDR |= 1u << (2u * SPI_INS.sck.pin);  // medium speed
    SPI_INS.mosi.port->OSPEEDR |= 1u << (2u * SPI_INS.mosi.pin);
    SPI2->CR1 |= SPI_CR1_SPE;
    SPI2->CR1 |= SPI_CR1_CSTART;
    ins_hw = true;
    ins_hw_timeouts = 0;
    ins_rx_reset();
}

static bool ins_reply(uint8_t want, uint8_t did, uint16_t offset, int32_t *status) {
    uint8_t type = ins_rx.buf[2] & 0x0Fu;
    if (want == ISB_ACK && (type == ISB_ACK || type == ISB_NACK)) {
        *status = type == ISB_ACK ? 1 : -2;
        return true;
    }
    if (want != ISB_DATA || type != ISB_DATA || ins_rx.buf[3] != did) return false;
    if ((ins_rx.buf[2] & ISB_FLAG_OFFSET) ? le16(&ins_rx.buf[6]) != offset : offset != 0u) return false;
    *status = 1;
    return true;
}

// Sends req (resending every resend_ms while the parser is between packets,
// for a module that may still be booting) and clocks the module until a reply
// matching want arrives. DATA payloads are copied to out. Returns the payload
// length (DATA), 1 (ACK), -2 (NACK) or -1 (timeout).
static int32_t ins_txn(const uint8_t *req, uint32_t req_len, uint32_t resend_ms, uint8_t want, uint8_t did,
                       uint16_t offset, uint8_t *out, uint32_t cap, uint32_t timeout_ms) {
    int32_t status = -1;
    uint32_t start = now_ms(), sent = 0;
    bool first = true;
    ins_select(true);
    while (status == -1 && (uint32_t)(now_ms() - start) < timeout_ms) {
        if (req && isb_idle(&ins_rx) && (first || (resend_ms && (uint32_t)(now_ms() - sent) >= resend_ms))) {
            // Full duplex: what arrives while the request goes out is parsed too.
            for (uint32_t i = 0; i < req_len && status == -1; i++) {
                if (isb_feed(&ins_rx, ins_byte(req[i]))) (void)ins_reply(want, did, offset, &status);
            }
            sent = now_ms();
            first = false;
            continue;
        }
        ins_sample_dr();
        if (isb_feed(&ins_rx, ins_byte(0))) (void)ins_reply(want, did, offset, &status);
    }
    if (status == 1 && want == ISB_DATA) {
        uint32_t off = (ins_rx.buf[2] & ISB_FLAG_OFFSET) ? 2u : 0u;
        uint32_t len = le16(&ins_rx.buf[4]) - off;
        if (len > cap) len = cap;
        for (uint32_t i = 0; i < len; i++) out[i] = ins_rx.buf[6u + off + i];
        status = (int32_t)len;
    }
    // Drain a few bytes so the packet tail (DR drops 1-2 bytes early) is out
    // before CS rises.
    for (int i = 0; i < 8; i++) (void)isb_feed(&ins_rx, ins_byte(0));
    ins_select(false);
    return status;
}

// One sample of a data set, or size bytes of it from offset (period 0).
static uint32_t ins_get(uint8_t did, uint16_t offset, uint16_t size, uint8_t *out, uint32_t cap, uint32_t timeout_ms) {
    // p_data_get_t: id, size (0 = whole set), offset, period (0 = once), all LE16.
    uint8_t body[8] = { did, 0, (uint8_t)size, (uint8_t)(size >> 8), (uint8_t)offset, (uint8_t)(offset >> 8), 0, 0 };
    uint8_t req[16];
    uint32_t len = isb_packet(req, ISB_GET_DATA, 0, 0, body, sizeof body);
    int32_t n = ins_txn(req, len, 250u, ISB_DATA, did, offset, out, cap, timeout_ms);
    return n > 0 ? (uint32_t)n : 0u;
}

// Writes n bytes of a data set at offset and waits for the ACK.
static int32_t ins_set(uint8_t did, uint16_t offset, const uint8_t *data, uint16_t n) {
    uint8_t req[150];
    if (n > sizeof req - 10u) return -1;
    uint32_t len = isb_packet(req, ISB_SET_DATA, did, offset, data, n);
    return ins_txn(req, len, 0, ISB_ACK, 0, 0, (uint8_t *)0, 0, 1000u);
}

// Starts (period multiple of the nav period) or stops a broadcast.
static void ins_stream(uint8_t did, uint16_t period) {
    uint8_t req[16];
    uint32_t len;
    if (period) {
        uint8_t body[8] = { did, 0, 0, 0, 0, 0, (uint8_t)period, (uint8_t)(period >> 8) };
        len = isb_packet(req, ISB_GET_DATA, 0, 0, body, sizeof body);
    } else {
        len = isb_packet(req, ISB_STOP_DID, did, 0, (const uint8_t *)0, 0);
    }
    ins_select(true);
    for (uint32_t i = 0; i < len; i++) (void)isb_feed(&ins_rx, ins_byte(req[i]));
    ins_select(false);
}

// Clocks out whatever the module still has queued: until DR has been low and
// the parser idle for 20 ms, or 500 ms.
static void ins_drain(void) {
    uint32_t start = now_ms(), quiet = now_ms();
    ins_select(true);
    while ((uint32_t)(now_ms() - start) < 500u && (uint32_t)(now_ms() - quiet) < 20u) {
        (void)isb_feed(&ins_rx, ins_byte(0));
        if (pin_read(INS_DRDY) || !isb_idle(&ins_rx)) quiet = now_ms();
    }
    ins_select(false);
}

static uint32_t isqrt(uint32_t v) {
    uint32_t r = 0;
    for (uint32_t bit = 1u << 30; bit; bit >>= 2) {
        if (v >= r + bit) {
            v -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
    }
    return r;
}

static bool ins_read_u32(uint8_t did, uint16_t offset, uint32_t *v) {
    uint8_t b[4];
    if (ins_get(did, offset, 4, b, sizeof b, 1000u) != sizeof b) return false;
    *v = le32(b);
    return true;
}

static bool ins_g9_drdy(void) {
    uint32_t io;
    return ins_read_u32(DID_FLASH_CONFIG, FLASH_CFG_IOCONFIG, &io) && (io & IO_CONFIG_G9_MASK) == IO_CONFIG_G9_SPI_DRDY;
}

// ---- Basic check (part of every run) ----

static void ins_checks(void) {
    // PC2/PC3 carry SPI2 MISO/MOSI only while their analog switches to
    // PC2_C/PC3_C (VMON_VIN, IMON_P5V_AUX) stay open, as they are at reset.
    RCC->APB4ENR |= RCC_APB4ENR_SYSCFGEN;
    (void)RCC->APB4ENR;
    uint32_t pmcr = SYSCFG->PMCR;
    const uint32_t so = SYSCFG_PMCR_PC2SO | SYSCFG_PMCR_PC3SO;
    check((pmcr & so) == so, "PC2/PC3 analog switches", "SYSCFG_PMCR 0x%08x (PC2SO and PC3SO set: open)", pmcr);

    ins_open_bitbang();
    ins_dr_seen = false;
    bool dr_idle = !pin_read(INS_DRDY);

    // The module boots on its own from P3V3_SENS1, so allow it boot time.
    uint8_t d[84];
    uint32_t n = ins_get(DID_DEV_INFO, 0, 0, d, sizeof d, 3000u);
    if (n == 0u && ins_rx.bad == 0u && !ins_dr_seen) {
        // Indistinguishable from an unfitted U26, which most boards are.
        warn("IMX-6 (SPI2, U26)", "no reply in 3 s and DR never high (unfitted, not booted in SPI mode, or no power)");
        return;
    }
    if (n < sizeof d) {
        fail("IMX-6 (SPI2, U26)", "%s: %u bytes of dev_info, %u bad packets, DR %s", n ? "short reply" : "no valid reply", n,
             ins_rx.bad, ins_dr_seen ? "seen high" : "never high");
        return;
    }
    char mfr[25], add[25];
    for (int i = 0; i < 24; i++) {
        mfr[i] = (char)d[28 + i];
        add[i] = (char)d[60 + i];
    }
    mfr[24] = add[24] = 0;
    // dev_info_t: hardwareType 2, hdwRunState 3, serialNumber 4, hardwareVer[3] 8,
    // hardwareVariant 11, firmwareVer 12, buildNumber 16, protocolVer 20.
    uint8_t type = d[2], state = d[3];
    check(type == IS_HDW_IMX && state == IS_RUN_APP, "IMX-6 (SPI2, U26)",
          "SN %u, hardware type %u %u.%u.%u variant %u, %s (expect type 3, application)", le32(&d[4]), type, d[8], d[9],
          d[10], d[11], state == IS_RUN_APP ? "application" : state == IS_RUN_BOOTLOADER ? "BOOTLOADER" : "run state unknown");
    info("IMX-6 firmware", "%u.%u.%u.%u build %u, protocol %u.%u.%u.%u, %s %u-%02u-%02u, %s", d[12], d[13], d[14], d[15],
         le32(&d[16]), d[20], d[21], d[22], d[23], mfr, 2000u + d[53], d[54], d[55], add);
    if (state != IS_RUN_APP) return;

    // sys_params_t: insStatus 4, hdwStatus 8, imuTemp 12, baroTemp 16, mcuTemp 20,
    // imuSamplePeriodMs 28, navOutputPeriodMs 32, genFaultCode 52.
    uint8_t s[64];
    if (ins_get(DID_SYS_PARAMS, 0, 0, s, sizeof s, 1000u) == sizeof s) {
        uint32_t hdw = le32(&s[8]);
        // The BASIC BIT run at boot fails on "EKF not initialized" (0x02000000)
        // without GNSS; only other AHRS-relevant BIT faults count here.
        uint8_t bt[56];
        bool bit_hw_fault = (hdw & HDW_STATUS_BIT_MASK) == HDW_STATUS_BIT_FAILED &&
                            (ins_get(DID_BIT, 0, 0, bt, sizeof bt, 1000u) != sizeof bt || (le32(&bt[4]) & 0xFDFF0F00u) != 0u);
        bool ok = (hdw & HDW_STATUS_SPI_ENABLED) && (hdw & HDW_STATUS_CRITICAL) == 0u && !bit_hw_fault &&
                  (hdw & HDW_STATUS_SATURATION) == 0u;
        check(ok, "IMX-6 status", "hdwStatus 0x%08x (SPI %s, BIT %s, %s, %s), insStatus 0x%08x, fault code 0x%x", hdw,
              (hdw & HDW_STATUS_SPI_ENABLED) ? "on" : "OFF",
              (hdw & HDW_STATUS_BIT_MASK) == HDW_STATUS_BIT_PASSED ? "passed"
              : (hdw & HDW_STATUS_BIT_MASK) != HDW_STATUS_BIT_FAILED ? "not run"
              : bit_hw_fault ? "FAILED" : "failed only on EKF not initialized (no GNSS)",
              (hdw & HDW_STATUS_CRITICAL) ? "CRITICAL FAULT" : "no critical fault",
              (hdw & HDW_STATUS_SATURATION) ? "SATURATED" : "no saturation", le32(&s[4]), le32(&s[52]));
        if (hdw & HDW_STATUS_CRITICAL) {
            uint8_t f[60];
            if (ins_get(DID_SYS_FAULT, 0, 0, f, sizeof f, 1000u) >= 32u) {
                info("IMX-6 last critical fault", "status 0x%08x at %u ms, file %u line %u, halt %u, pc 0x%08x, lr 0x%08x, psr 0x%08x",
                     le32(&f[4]), le32(&f[0]), le32(&f[8]), le32(&f[12]), le32(&f[16]), le32(&f[24]), le32(&f[20]), le32(&f[28]));
            }
        }
        info("IMX-6 temperatures", "IMU %d C, baro %d C, MCU %d C; IMU period %u ms, nav period %u ms",
             to_int(lef(&s[12]), 999.0f), to_int(lef(&s[16]), 999.0f), to_int(lef(&s[20]), 999.0f), le32(&s[28]),
             le32(&s[32]));
    } else {
        fail("IMX-6 status", "no DID_SYS_PARAMS reply");
    }

    // One preintegrated IMU sample. pimu_t: time (double) 0, dt 8, status 12,
    // theta[3] 16 (rad), vel[3] 28 (m/s), body frame.
    uint8_t p[40];
    n = ins_get(DID_PIMU, 0, 0, p, sizeof p, 1000u);
    float dt = n == sizeof p ? lef(&p[8]) : 0.0f;
    if (!(dt > 0.0f && dt < 1.0f)) {
        fail("IMX-6 IMU sample", n == sizeof p ? "dt %d us out of range" : "no PIMU reply in 1 s", to_int(dt * 1e6f, 1e9f));
        return;
    }
    int32_t a[3], g[3];
    uint32_t a2 = 0, g2 = 0;
    for (int i = 0; i < 3; i++) {
        a[i] = to_int(lef(&p[28 + 4 * i]) / dt * (1000.0f / 9.80665f), 20000.0f);  // mg
        g[i] = to_int(lef(&p[16 + 4 * i]) / dt * 57295.78f, 200000.0f);             // mdeg/s
        a2 += (uint32_t)(a[i] * a[i]);
        g2 += (uint32_t)(g[i] / 10 * (g[i] / 10));
    }
    uint32_t am = isqrt(a2), gm = isqrt(g2) * 10u;
    check(am > 900u && am < 1100u, "IMX-6 accelerometer", "%d, %d, %d mg, |a| %u mg at rest (expect 900-1100), dt %u us",
          a[0], a[1], a[2], am, (uint32_t)to_int(dt * 1e6f, 1e6f));
    info("IMX-6 gyro", "%d, %d, %d mdeg/s, |w| %u mdeg/s; status 0x%08x", g[0], g[1], g[2], gm, le32(&p[12]));

    // magnetometer_t: mag[3] at 8 (uT). barometer_t: bar 8 (kPa), barTemp 16 (C).
    uint8_t m[20];
    if (ins_get(DID_MAGNETOMETER, 0, 0, m, sizeof m, 1000u) == sizeof m) {
        float x = lef(&m[8]), y = lef(&m[12]), z = lef(&m[16]);
        int32_t mm = to_int(fsqrt(x * x + y * y + z * z) * 10.0f, 1e6f);
        check(mm > 0, "IMX-6 magnetometer", "%d, %d, %d uT, |B| %d.%d uT", to_int(x, 1e5f), to_int(y, 1e5f), to_int(z, 1e5f),
              mm / 10, mm % 10);
    } else {
        fail("IMX-6 magnetometer", "no DID_MAGNETOMETER reply");
    }
    uint8_t b[24];
    if (ins_get(DID_BAROMETER, 0, 0, b, sizeof b, 1000u) == sizeof b) {
        int32_t pa = to_int(lef(&b[8]) * 1000.0f, 1e7f);
        check(pa > 70000 && pa < 110000, "IMX-6 barometer", "%d Pa (expect 70-110 kPa), %d C", pa, to_int(lef(&b[16]), 999.0f));
    } else {
        fail("IMX-6 barometer", "no DID_BAROMETER reply");
    }

    // G9 is data ready only with DID_FLASH_CONFIG.ioConfig bits 5:4 set to
    // IO_CONFIG_G9_SPI_DRDY; the factory default (0) puts data ready on G13,
    // which this board leaves unconnected.
    uint32_t io_cfg;
    if (!ins_read_u32(DID_FLASH_CONFIG, FLASH_CFG_IOCONFIG, &io_cfg)) {
        fail("IMX-6 ioConfig", "no reply to a 4-byte DID_FLASH_CONFIG read at offset %u", FLASH_CFG_IOCONFIG);
        return;
    }
    if ((io_cfg & IO_CONFIG_G9_MASK) != IO_CONFIG_G9_SPI_DRDY) {
        warn("IMX-6 data ready (PA6)", "G9 data ready not enabled: ioConfig 0x%08x, bits 5:4 = %u (needs 3; set it over UART0 with EvalTool: writing it over SPI crashes firmware 3.1.1.0)",
             io_cfg, (io_cfg & IO_CONFIG_G9_MASK) >> 4);
        return;
    }

    // DR is high while a reply waits and low once it has been read.
    ins_drain();
    dr_idle = !pin_read(INS_DRDY);
    ins_dr_seen = false;
    (void)ins_get(DID_DEV_INFO, 0, 0, d, sizeof d, 1000u);
    bool dr_after = !pin_read(INS_DRDY);
    check(dr_idle && ins_dr_seen && dr_after, "IMX-6 data ready (PA6)", "%s before, %s with a reply waiting, %s after",
          dr_idle ? "low" : "HIGH", ins_dr_seen ? "high" : "NEVER HIGH", dr_after ? "low" : "HIGH");
}

static void test_ins(void) {
    ins_checks();
    ins_release();
}

// ---- 'g': enable G9 data ready in the module's flash configuration ----
//
// Mirrors the SDK's ISDevice::SetImxFlashConfig: send the new checksum, then the
// changed field, and wait for sys_params.flashCfgChecksum to match once the
// module has saved. The checksum is an XOR of 32-bit words, so changing one
// word changes it by old ^ new.

static double ins_uptime(void) {
    uint8_t p[40];
    return ins_get(DID_PIMU, 0, 0, p, sizeof p, 2000u) == sizeof p ? led(p) : -1.0;
}

// Writes ioConfig (with the checksum, in one SET_DATA) and waits for the save.
// Returns true once the module reports the new checksum and reads it back.
static bool ins_write_word(uint16_t word, uint32_t (*change)(uint32_t), const char *name) {
    uint32_t ck, io;
    if (!ins_read_u32(DID_FLASH_CONFIG, FLASH_CFG_CHECKSUM, &ck) || !ins_read_u32(DID_FLASH_CONFIG, word, &io)) {
        fail(name, "no flash config reply over SPI2");
        return false;
    }
    uint32_t io_new = change(io), ck_new = ck ^ io ^ io_new;
    if (io_new == io) {
        pass(name, "ioConfig already 0x%08x", io);
        return true;
    }
    // One SET_DATA from the checksum through ioConfig (132 bytes read back
    // first), so the module never holds a checksum that doesn't match.
    static uint8_t blk[FLASH_CFG_IOCONFIG + 4u - FLASH_CFG_CHECKSUM];
    uint32_t got = ins_get(DID_FLASH_CONFIG, FLASH_CFG_CHECKSUM, sizeof blk, blk, sizeof blk, 1000u);
    if (got != sizeof blk || le32(blk) != ck || le32(&blk[word - FLASH_CFG_CHECKSUM]) != io) {
        fail(name, "read of %u bytes at offset %u came back %u bytes or changed; nothing written", (uint32_t)sizeof blk,
             FLASH_CFG_CHECKSUM, got);
        return false;
    }
    put32(blk, ck_new);
    put32(&blk[word - FLASH_CFG_CHECKSUM], io_new);
    double t0 = ins_uptime();
    int32_t r = ins_set(DID_FLASH_CONFIG, FLASH_CFG_CHECKSUM, blk, sizeof blk);
    uint32_t start = now_ms(), saved = 0;
    while ((uint32_t)(now_ms() - start) < 6000u) {
        delay_ms(500);
        if (ins_read_u32(DID_SYS_PARAMS, 44, &saved) && saved == ck_new) break;
    }
    double t1 = ins_uptime();
    uint32_t ck_back = 0, io_back = 0;
    bool back = ins_read_u32(DID_FLASH_CONFIG, FLASH_CFG_CHECKSUM, &ck_back) &&
                ins_read_u32(DID_FLASH_CONFIG, word, &io_back);
    bool ok = back && io_back == io_new && ck_back == ck_new && saved == ck_new;
    check(ok, name, "word %u: 0x%08x -> 0x%08x (%s): read back 0x%08x, saved checksum %s; module %s (uptime %d -> %d ms)", word, io,
          io_new, r == 1 ? "ACK" : r == -2 ? "NACK" : "no reply", io_back, saved == ck_new ? "matches" : "OLD",
          t1 > t0 ? "kept running" : "RESET", to_int((float)(t0 * 1000.0), 2e9f), to_int((float)(t1 * 1000.0), 2e9f));
    return ok;
}

static uint32_t io_g9_drdy(uint32_t io) {
    return (io & ~IO_CONFIG_G9_MASK) | IO_CONFIG_G9_SPI_DRDY;
}
static uint32_t io_strobe_flip(uint32_t io) {
    return io ^ 0x1u;  // IO_CONFIG_STROBE_TRIGGER_HIGH: strobe edge polarity, nothing here uses it
}

static void ins_enable_drdy(void) {
    passes = fails = warnings = 0;
    con_puts("\n=== IMX-6: enable G9 data ready ===\n");
    ins_open_bitbang();
    (void)ins_write_word(FLASH_CFG_IOCONFIG, io_g9_drdy, "IMX-6 ioConfig G9 data ready");
    ins_release();
    con_printf("=== %u passed, %u failed, %u warnings ===\n", passes, fails, warnings);
}

// ---- 'x': does a SET_DATA over SPI survive? (module uptime from PIMU) ----

static void ins_set_probe(const char *name, uint8_t did, uint16_t offset, const uint8_t *data, uint16_t n) {
    double t0 = ins_uptime();
    int32_t r = ins_set(did, offset, data, n);
    delay_ms(300);
    double t1 = ins_uptime();
    check(t0 >= 0.0 && t1 > t0, name, "%s; uptime %d ms before, %d ms after", r == 1 ? "ACK" : r == -2 ? "NACK" : "no reply",
          to_int((float)(t0 * 1000.0), 2e9f), to_int((float)(t1 * 1000.0), 2e9f));
}

static void ins_set_tests(void) {
    passes = fails = warnings = 0;
    con_puts("\n=== IMX-6 SET_DATA over SPI ===\n");
    ins_open_bitbang();
    uint8_t cmd[8];
    put32(cmd, 0x1234u);
    put32(cmd + 4, 0u);  // command and inverse don't match: the module must ignore it
    ins_set_probe("SET DID_SYS_CMD (ignored pair)", DID_SYS_CMD, 0, cmd, sizeof cmd);
    uint32_t v;
    if (ins_read_u32(DID_FLASH_CONFIG, 12, &v)) {  // startupImuDtMs, written back unchanged
        uint8_t b[4];
        put32(b, v);
        ins_set_probe("SET DID_FLASH_CONFIG startupImuDtMs (same value)", DID_FLASH_CONFIG, 12, b, 4);
    }
    // A real flash save of an unrelated ioConfig bit, then back.
    // A real flash save of a field other than ioConfig, then back:
    // lastLlaUpdateDistance (offset 128, float m) nudged by its lowest bit.
    if (ins_write_word(128, io_strobe_flip, "Flash save: lastLlaUpdateDistance LSB flipped")) {
        (void)ins_write_word(128, io_strobe_flip, "Flash save: lastLlaUpdateDistance restored");
    }
    ins_release();
    con_printf("=== %u passed, %u failed, %u warnings ===\n", passes, fails, warnings);
}

// ---- 's': live IMX-6 data over SPI2 (4 MHz, polled) until a key is pressed ----
//
// Streams PIMU and INS_1 at the nav period (200 Hz) and the magnetometer and
// barometer at 10 Hz, and prints the latest of each at 10 Hz.

static void ins_live(void) {
    con_puts("\n=== IMX-6 live data (any key stops) ===\n");
    con_puts("time ms | accel mg x y z | gyro mdeg/s x y z | roll pitch yaw 0.1 deg | mag uT x y z | baro Pa | PIMU/INS_1 Hz | bad\n");
    ins_open_hw(3);
    ins_stream(DID_PIMU, 1);
    ins_stream(DID_INS_1, 1);
    ins_stream(DID_MAGNETOMETER, 20);
    ins_stream(DID_BAROMETER, 20);
    int32_t a[3] = { 0 }, g[3] = { 0 }, e[3] = { 0 }, m[3] = { 0 }, pa = 0, t_ms = 0;
    uint32_t n_pimu = 0, n_ins = 0, last = now_ms();
    ins_select(true);
    while (!(CONSOLE->ISR & USART_ISR_RXNE_RXFNE)) {
        if (isb_feed(&ins_rx, ins_byte(0)) && (ins_rx.buf[2] & 0x0Fu) == ISB_DATA) {
            const uint8_t *p = &ins_rx.buf[6];
            uint16_t len = le16(&ins_rx.buf[4]);
            if (ins_rx.buf[3] == DID_PIMU && len == 40u) {
                float dt = lef(&p[8]);
                if (dt > 0.0f && dt < 1.0f) {
                    for (int i = 0; i < 3; i++) {
                        a[i] = to_int(lef(&p[28 + 4 * i]) / dt * (1000.0f / 9.80665f), 1e6f);
                        g[i] = to_int(lef(&p[16 + 4 * i]) / dt * 57295.78f, 1e7f);
                    }
                }
                t_ms = to_int((float)(led(p) * 1000.0), 2e9f);
                n_pimu++;
            } else if (ins_rx.buf[3] == DID_INS_1 && len == 80u) {
                for (int i = 0; i < 3; i++) e[i] = to_int(lef(&p[20 + 4 * i]) * 572.9578f, 1e6f);  // theta rad -> 0.1 deg
                n_ins++;
            } else if (ins_rx.buf[3] == DID_MAGNETOMETER && len == 20u) {
                for (int i = 0; i < 3; i++) m[i] = to_int(lef(&p[8 + 4 * i]), 1e5f);
            } else if (ins_rx.buf[3] == DID_BAROMETER && len == 24u) {
                pa = to_int(lef(&p[8]) * 1000.0f, 1e7f);
            }
        }
        uint32_t now = now_ms();
        if ((uint32_t)(now - last) >= 100u) {
            uint32_t span = now - last;
            last = now;
            con_printf("%d | %d %d %d | %d %d %d | %d %d %d | %d %d %d | %d | %u/%u | %u\n", t_ms, a[0], a[1], a[2], g[0], g[1],
                       g[2], e[0], e[1], e[2], m[0], m[1], m[2], pa, n_pimu * 1000u / span, n_ins * 1000u / span, ins_rx.bad);
            n_pimu = n_ins = 0;
        }
    }
    (void)CONSOLE->RDR;
    ins_select(false);
    ins_stream(DID_PIMU, 0);
    ins_stream(DID_INS_1, 0);
    ins_stream(DID_MAGNETOMETER, 0);
    ins_stream(DID_BAROMETER, 0);
    ins_drain();
    ins_release();
    con_puts("=== stopped ===\n");
}

// ---- 'i': IMX-6 deep test ----

typedef struct {
    uint32_t pkts[2], other, gaps, missing;
    double t_last;
    float a0[3], w0[3], sa[3], sa2[3], sw[3], sw2[3];
    uint32_t n;
} InsStream;

static void stream_pimu(InsStream *st, const uint8_t *p) {
    double t = led(&p[0]);
    float dt = lef(&p[8]);
    if (!(dt > 0.0f && dt < 1.0f)) return;
    if (st->n) {
        float gap = (float)(t - st->t_last);
        if (gap > 1.5f * dt) {
            st->gaps++;
            st->missing += (uint32_t)to_int(gap / dt + 0.5f, 1e6f) - 1u;
        }
    }
    st->t_last = t;
    for (int i = 0; i < 3; i++) {
        float a = lef(&p[28 + 4 * i]) / dt * (1000.0f / 9.80665f);  // mg
        float w = lef(&p[16 + 4 * i]) / dt * 57295.78f;            // mdeg/s
        if (!st->n) {
            st->a0[i] = a;
            st->w0[i] = w;
        }
        a -= st->a0[i];  // shifted sums keep float precision
        w -= st->w0[i];
        st->sa[i] += a;
        st->sa2[i] += a * a;
        st->sw[i] += w;
        st->sw2[i] += w * w;
    }
    st->n++;
}

// Streams PIMU (and INS_1 when both) at the nav period for ms. DR-gated when
// G9 data ready is enabled: only clocks while DR is high or a packet is open.
static void ins_stream_test(const char *name, uint32_t ms, bool both, bool gated, uint32_t nav_ms) {
    InsStream st = { 0 };
    ins_rx_reset();
    ins_dr_rises = 0;
    ins_dr_last = pin_read(INS_DRDY);
    ins_stream(DID_PIMU, 1);
    if (both) ins_stream(DID_INS_1, 1);
    uint32_t start = now_ms(), clocked = 0;
    ins_select(true);
    while ((uint32_t)(now_ms() - start) < ms) {
        ins_sample_dr();
        if (gated && !ins_dr_last && isb_idle(&ins_rx)) continue;
        clocked++;
        if (!isb_feed(&ins_rx, ins_byte(0))) continue;
        uint8_t type = ins_rx.buf[2] & 0x0Fu, id = ins_rx.buf[3];
        if (type == ISB_DATA && id == DID_PIMU && le16(&ins_rx.buf[4]) == 40u) {
            st.pkts[0]++;
            stream_pimu(&st, &ins_rx.buf[6]);
        } else if (type == ISB_DATA && id == DID_INS_1) {
            st.pkts[1]++;
        } else {
            st.other++;
        }
    }
    ins_select(false);
    ins_stream(DID_PIMU, 0);
    if (both) ins_stream(DID_INS_1, 0);
    ins_drain();

    uint32_t expect = nav_ms ? ms / nav_ms : 0u;
    bool ok = st.pkts[0] + 2u >= expect && ins_rx.bad == 0u && st.gaps == 0u && (!both || st.pkts[1] + 2u >= expect) &&
              (!ins_hw || ins_hw_timeouts == 0u);
    check(ok, name, "PIMU %u, INS_1 %u, of %u expected each in %u ms, %u bad, %u gaps (%u missing), %u other, DR rises %u, %u bytes",
          st.pkts[0], st.pkts[1], expect, ms, ins_rx.bad, st.gaps, st.missing, st.other, ins_dr_rises, clocked);
    if (ins_hw && ins_hw_timeouts) info("  SPI2", "%u byte timeouts, SR 0x%08x", ins_hw_timeouts, SPI2->SR);
    if (st.n < 2u) return;
    int32_t am[3], as[3], wm[3], ws[3];
    for (int i = 0; i < 3; i++) {
        float n = (float)st.n, ma = st.sa[i] / n, mw = st.sw[i] / n;
        am[i] = to_int(st.a0[i] + ma, 1e6f);
        as[i] = to_int(fsqrt(st.sa2[i] / n - ma * ma), 1e6f);
        wm[i] = to_int(st.w0[i] + mw, 1e7f);
        ws[i] = to_int(fsqrt(st.sw2[i] / n - mw * mw), 1e7f);
    }
    info("  IMU over the stream", "accel mean %d, %d, %d mg, sd %d, %d, %d mg; gyro mean %d, %d, %d mdeg/s, sd %d, %d, %d mdeg/s",
         am[0], am[1], am[2], as[0], as[1], as[2], wm[0], wm[1], wm[2], ws[0], ws[1], ws[2]);
}

// Hammers one-shot requests: count of good DEV_INFO replies out of n.
static void ins_request_test(const char *name, uint32_t n) {
    uint8_t d[84];
    uint32_t good = 0, start = now_ms();
    ins_rx_reset();
    for (uint32_t i = 0; i < n; i++) {
        if (ins_get(DID_DEV_INFO, 0, 0, d, sizeof d, 200u) == sizeof d && d[2] == IS_HDW_IMX) good++;
    }
    check(good == n && ins_rx.bad == 0u && (!ins_hw || ins_hw_timeouts == 0u), name,
          "%u of %u DEV_INFO replies, %u bad packets, %u ms", good, n, ins_rx.bad, (uint32_t)(now_ms() - start));
}

// Time from now until DEV_INFO answers (bit-banged), or 0 after timeout_ms.
static uint32_t ins_boot_time(uint32_t t0, uint32_t timeout_ms) {
    uint8_t d[84];
    ins_open_bitbang();
    while ((uint32_t)(now_ms() - t0) < timeout_ms) {
        if (ins_get(DID_DEV_INFO, 0, 0, d, sizeof d, 100u) == sizeof d) return now_ms() - t0;
    }
    return 0;
}

static void ins_reset_report(const char *name, uint32_t boot_ms, uint32_t expect_cause) {
    uint32_t hdw = 0, t = now_ms();
    bool got = false;
    while (boot_ms && (uint32_t)(now_ms() - t) < 5000u) {  // sys_params reads all zero for a while after boot
        got = ins_read_u32(DID_SYS_PARAMS, 8, &hdw) && hdw != 0u;
        if (got) break;
        delay_ms(100);
    }
    bool drdy = got && ins_g9_drdy();
    check(got && (hdw & HDW_STATUS_SPI_ENABLED) && (!expect_cause || (hdw & HDW_STATUS_RESET_CAUSE) == expect_cause), name,
          boot_ms ? "answered %u ms after, hdwStatus 0x%08x (SPI %s, reset cause %u), G9 data ready %s"
                  : "no DEV_INFO reply within 10 s",
          boot_ms, hdw, (hdw & HDW_STATUS_SPI_ENABLED) ? "on" : "OFF", (hdw & HDW_STATUS_RESET_CAUSE) >> 28,
          drdy ? "on" : "off");
}

static void ins_deep(void) {
    passes = fails = warnings = 0;
    leds(false, false, true);
    con_puts("\n=== IMX-6 deep test ===\n");
    ins_checks();
    uint8_t d0[84];
    if (ins_get(DID_DEV_INFO, 0, 0, d0, sizeof d0, 1000u) != sizeof d0) {
        ins_release();
        con_printf("=== %u passed, %u failed, %u warnings ===\n", passes, fails, warnings);
        leds(true, false, false);
        return;
    }

    // The module's own built-in test: the result of the one run at boot, then a
    // full stationary run (keep the board still).
    uint8_t bt[56];
    if (ins_get(DID_BIT, 0, 0, bt, sizeof bt, 1000u) == sizeof bt) {
        info("IMX-6 BIT at boot", "state %u, hdwBitStatus 0x%08x, calBitStatus 0x%08x", bt[2], le32(&bt[4]), le32(&bt[8]));
    }
    uint8_t bit_cmd = 2;  // BIT_CMD_FULL_STATIONARY
    int32_t r = ins_set(DID_BIT, 0, &bit_cmd, 1);
    uint32_t t_bit = now_ms();
    bool done = false;
    while (!done && (uint32_t)(now_ms() - t_bit) < 30000u) {
        delay_ms(500);
        done = ins_get(DID_BIT, 0, 0, bt, sizeof bt, 1000u) == sizeof bt && bt[2] == 1u && bt[1] == bit_cmd;
    }
    uint32_t hb = le32(&bt[4]), cb = le32(&bt[8]);
    check(done && (hb & 0x0Fu) != 0u && (hb & 0xFFFF0F00u) == 0u && (cb & 0x0Fu) != 0u && (cb & 0x00FFFF00u) == 0u,
          "IMX-6 BIT (full, stationary)",
          "%s after %u ms: hdwBitStatus 0x%08x, calBitStatus 0x%08x, gyro error %d mdeg/s (sd %d), accel error %d mg (sd %d)",
          r != 1 ? "command not ACKed" : done ? "done" : "NOT DONE", (uint32_t)(now_ms() - t_bit), hb, cb,
          to_int(lef(&bt[36]) * 57295.78f, 1e7f), to_int(lef(&bt[44]) * 57295.78f, 1e7f),
          to_int(lef(&bt[40]) * (1000.0f / 9.80665f), 1e7f), to_int(lef(&bt[48]) * (1000.0f / 9.80665f), 1e7f));

    uint32_t nav_ms = 0;
    (void)ins_read_u32(DID_SYS_PARAMS, 32, &nav_ms);
    bool gated = ins_g9_drdy();

    ins_open_bitbang();
    ins_request_test("Bit-bang 200 kHz requests", 20);
    ins_stream_test(gated ? "Bit-bang 200 kHz PIMU stream, DR-gated" : "Bit-bang 200 kHz PIMU stream, polled", 3000u, false,
                    gated, nav_ms);

    static const struct {
        uint32_t mbr;
        const char *req, *stream;
    } speeds[] = {
        { 5, "SPI2 1 MHz requests", "SPI2 1 MHz PIMU + INS_1 stream" },
        { 4, "SPI2 2 MHz requests", "SPI2 2 MHz PIMU + INS_1 stream" },
        { 3, "SPI2 4 MHz requests", "SPI2 4 MHz PIMU + INS_1 stream" },
    };
    for (unsigned i = 0; i < sizeof speeds / sizeof speeds[0]; i++) {
        ins_open_hw(speeds[i].mbr);
        ins_request_test(speeds[i].req, 50);
        ins_stream_test(speeds[i].stream, 3000u, true, gated, nav_ms);
    }
    // Beyond the 5 Mbit/s the spec allows: margin only, never a failure.
    ins_open_hw(2);
    uint8_t d[84];
    uint32_t good = 0;
    for (int i = 0; i < 50; i++) good += ins_get(DID_DEV_INFO, 0, 0, d, sizeof d, 200u) == sizeof d;
    info("SPI2 8 MHz requests (beyond spec)", "%u of 50 DEV_INFO replies, %u bad packets", good, ins_rx.bad);
    ins_drain();

    // Software reset (DID_SYS_CMD), then three power cycles of P3V3_SENS1 with
    // the SPI2 lines floating, each timed to the first DEV_INFO reply.
    ins_open_bitbang();
    uint8_t cmd[8];
    put32(cmd, SYS_CMD_SOFTWARE_RESET);
    put32(cmd + 4, ~SYS_CMD_SOFTWARE_RESET);
    (void)ins_set(DID_SYS_CMD, 0, cmd, sizeof cmd);
    uint32_t t0 = now_ms();
    delay_ms(50);
    ins_reset_report("IMX-6 software reset", ins_boot_time(t0, 10000u), 0x30000000u);

    uint32_t worst = 0;
    for (int i = 0; i < 3; i++) {
        ins_release();
        pin_output(SENS1_EN, false, false);
        delay_ms(200);
        pin_input(SENS1_EN, PullNone);  // back to the R60 pull-up
        t0 = now_ms();
        uint32_t boot = ins_boot_time(t0, 10000u);
        if (boot > worst) worst = boot;
        ins_reset_report(i == 0 ? "IMX-6 power cycle 1" : i == 1 ? "IMX-6 power cycle 2" : "IMX-6 power cycle 3", boot, 0);
    }
    info("IMX-6 boot time", "%u ms worst case from P3V3_SENS1 on to the first DEV_INFO reply", worst);
    ins_checks();  // everything still answers after the cycles

    ins_release();
    con_printf("=== %u passed, %u failed, %u warnings ===\n", passes, fails, warnings);
    leds(fails != 0u, fails == 0u, false);
}

// ---- ADC: VIN, VBAT, P5V_AUX current, VDDA ----

static bool adc_up(ADC_TypeDef *adc, ADC_Common_TypeDef *common, uint32_t ccr_extra, bool rev_v) {
    if ((adc->CR & ADC_CR_ADEN) != 0u) return true;
    // Synchronous clock HCLK/4 = 16 MHz, set while the ADCs are disabled.
    common->CCR = (common->CCR & ~ADC_CCR_CKMODE) | ADC_CCR_CKMODE_1 | ADC_CCR_CKMODE_0 | ccr_extra;
    adc->CR &= ~ADC_CR_DEEPPWD;
    adc->CR |= ADC_CR_ADVREGEN;
    delay_us(20);  // LL_ADC_DELAY_INTERNAL_REGUL_STAB_US is 10
    // Boost per ST HAL ADC_ConfigureBoostMode: rev V halves the clock (8 MHz:
    // BOOST = 01), rev Y sets BOOST_0 only above 20 MHz.
    adc->CR = (adc->CR & ~ADC_CR_BOOST) | (rev_v ? ADC_CR_BOOST_0 : 0u);
    adc->CR &= ~ADC_CR_ADCALDIF;
    adc->CR |= ADC_CR_ADCALLIN;
    adc->CR |= ADC_CR_ADCAL;
    if (!wait_flag(&adc->CR, ADC_CR_ADCAL, false, 100u)) return false;
    adc->ISR = ADC_ISR_ADRDY;
    adc->CR |= ADC_CR_ADEN;
    if (!wait_flag(&adc->ISR, ADC_ISR_ADRDY, true, 10u)) return false;
    adc->CFGR = ADC_CFGR_OVRMOD;  // 16-bit, single conversion, software trigger
    return true;
}

static uint32_t adc_read(ADC_TypeDef *adc, uint32_t ch) {
    adc->PCSEL |= 1u << ch;
    if (ch < 10u) adc->SMPR1 = (adc->SMPR1 & ~(7u << (3u * ch))) | (7u << (3u * ch));
    else adc->SMPR2 = (adc->SMPR2 & ~(7u << (3u * (ch - 10u)))) | (7u << (3u * (ch - 10u)));
    adc->SQR1 = ch << ADC_SQR1_SQ1_Pos;  // one conversion
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) {
        adc->ISR = ADC_ISR_EOC;
        adc->CR |= ADC_CR_ADSTART;
        if (!wait_flag(&adc->ISR, ADC_ISR_EOC, true, 10u)) return 0;
        sum += adc->DR & 0xFFFFu;
    }
    return sum / 16u;
}

static void test_adc(void) {
    bool rev_v = (DBGMCU->IDCODE >> 16) == 0x2003u;
    RCC->AHB1ENR |= RCC_AHB1ENR_ADC12EN;
    RCC->AHB4ENR |= RCC_AHB4ENR_ADC3EN;
    (void)RCC->AHB4ENR;
    bool adc1 = adc_up(ADC1, ADC12_COMMON, 0, rev_v);
    bool adc3 = adc_up(ADC3, ADC3_COMMON, ADC_CCR_VREFEN | ADC_CCR_TSEN | ADC_CCR_VBATEN, rev_v);
    check(adc1 && adc3, "ADC1/ADC3", "calibrated and ready");
    if (!adc1 || !adc3) return;

    delay_us(10);  // VREFINT stabilisation (LL: 5 us)
    uint32_t vref_raw = adc_read(ADC3, ADC_CH_VREFINT);
    uint32_t cal = *(volatile uint16_t *)0x1FF1E860u;  // VREFINT_CAL, 16-bit raw at 3.3 V
    uint32_t vdda_mv = vref_raw ? (cal * 3300u) / vref_raw : 0u;
    check(vdda_mv > 3100u && vdda_mv < 3500u, "VDDA", "%u mV (VREFINT raw %u, cal %u)", vdda_mv, vref_raw, cal);

    uint32_t vin_raw = adc_read(ADC3, ADC_CH_VIN);
    uint32_t vin_mv = (uint32_t)(((uint64_t)vin_raw * vdda_mv * DIVIDER_NUM) / (65535u * DIVIDER_DEN));
    check(vin_mv > 4500u && vin_mv < 21000u, "VIN (PC2_C)", "%u mV (USB-C 5-20 V or 4S battery)", vin_mv);

    uint32_t vbat_raw = adc_read(ADC1, ADC_CH_VBAT);
    uint32_t vbat_mv = (uint32_t)(((uint64_t)vbat_raw * vdda_mv * DIVIDER_NUM) / (65535u * DIVIDER_DEN));
    if (vbat_mv < 1000u) info("VBAT (PA0_C)", "%u mV (no battery on J12)", vbat_mv);
    else check(vbat_mv > 11000u && vbat_mv < 17500u, "VBAT (PA0_C)", "%u mV (4S: 12.0-16.8 V)", vbat_mv);

    uint32_t imon_raw = adc_read(ADC3, ADC_CH_IMON);
    uint32_t imon_mv = (uint32_t)(((uint64_t)imon_raw * vdda_mv) / 65535u);
    info("P5V_AUX current (PC3_C)", "%u mA (IMON %u mV at 0.200 V/A)", imon_mv * 5u, imon_mv);

    delay_us(30);  // LL_ADC_DELAY_TEMPSENSOR_STAB_US is 26
    int32_t ts_raw = (int32_t)((adc_read(ADC3, ADC_CH_TEMP) * vdda_mv) / 3300u);  // to the 3.3 V calibration
    int32_t cal1 = *(volatile uint16_t *)TS_CAL1_ADDR, cal2 = *(volatile uint16_t *)TS_CAL2_ADDR;
    int32_t temp_c = cal2 != cal1 ? 30 + (ts_raw - cal1) * (130 - 30) / (cal2 - cal1) : -999;
    check(temp_c > 0 && temp_c < 85, "Die temperature", "%d C (raw %d, TS_CAL1 %d at 30 C, TS_CAL2 %d at 130 C)", temp_c,
          ts_raw, cal1, cal2);

    uint32_t cell_mv = (uint32_t)(((uint64_t)adc_read(ADC3, ADC_CH_VBAT_INT) * vdda_mv * 4u) / 65535u);
    if (cell_mv < 2000u) warn("RTC backup cell (BT1)", "%u mV: flat or missing (MS421R works from 2.0 V)", cell_mv);
    else check(cell_mv < 3400u, "RTC backup cell (BT1)", "%u mV (MS421R: 2.0-3.3 V)", cell_mv);
}

// ---- CAN transceivers: TXD dominant must come back on RXD ----

static void can_echo(const char *name, Pin tx, Pin rx, Pin other_rx, const char *other) {
    pin_input(rx, PullNone);
    pin_input(other_rx, PullNone);
    pin_output(tx, true, false);
    delay_us(50);
    bool recessive = pin_read(rx);
    pin_write(tx, false);  // dominant for 20 us, far below the TJA1051 TXD time-out
    delay_us(10);
    bool dominant = !pin_read(rx);
    bool other_quiet = pin_read(other_rx);
    delay_us(10);
    pin_write(tx, true);
    delay_us(10);
    bool released = pin_read(rx);
    pin_input(tx, PullNone);
    check(recessive && dominant && released, name, "RXD %s with TXD high, %s with TXD low",
          recessive && released ? "high" : "LOW", dominant ? "low (bus dominant)" : "HIGH (no echo)");
    check(other_quiet, name, "%s stays recessive while this bus is dominant", other);
}

static void test_can(void) {
    can_echo("CAN1 (U13, PH13/PH14)", CAN1_TX, CAN1_RX, CAN2_RX, "CAN2");
    can_echo("CAN2 (U12, PB6/PB5)", CAN2_TX, CAN2_RX, CAN1_RX, "CAN1");
}

// ---- UART6: the probe's second console, checked against a host responder ----

static void test_uart6(void) {
    RCC->APB2ENR |= RCC_APB2ENR_USART6EN;
    (void)RCC->APB2ENR;
    uart_init(USART6, UART6_TX, UART6_RX, UART6_AF, 115200);
    delay_us(100);
    bool idle = pin_read(UART6_RX);
    uart_drain(USART6);
    const char *ping = "Aleph FC UART6 ping\r\n";
    for (const char *s = ping; *s; s++) uart_putc(USART6, *s);
    char got[8] = { 0 };
    int n = 0;
    uint32_t start = now_ms();
    while ((uint32_t)(now_ms() - start) < 2000u) {
        int c = uart_getc(USART6);
        if (c < 0) continue;
        got[n % 4] = (char)c;  // last four characters
        n++;
        if (n >= 4) {
            char last[5];
            for (int i = 0; i < 4; i++) last[i] = got[(n + i) % 4];
            last[4] = 0;
            if (last[0] == 'p' && last[1] == 'o' && last[2] == 'n' && last[3] == 'g') break;
        }
    }
    bool ponged = false;
    if (n >= 4) {
        char last[4];
        for (int i = 0; i < 4; i++) last[i] = got[(n + i) % 4];
        ponged = last[0] == 'p' && last[1] == 'o' && last[2] == 'n' && last[3] == 'g';
    }
    if (ponged) pass("UART6 (PC6/PC7)", "ping sent, pong received through the probe's UART6 console");
    else warn("UART6 (PC6/PC7)", "RX idle %s; no pong in 2 s (needs a host answering 'pong' on the UART6 console)",
              idle ? "high" : "LOW");
    pin_input(UART6_TX, PullNone);
}

// ---- Board-ID EEPROM: write, read back and restore the last byte ----

static bool eeprom_addr(uint16_t addr) {
    i2c_start(&I2C_EEPROM);
    return i2c_write_byte(&I2C_EEPROM, EEPROM_ADDR << 1) && i2c_write_byte(&I2C_EEPROM, (uint8_t)(addr >> 8)) &&
           i2c_write_byte(&I2C_EEPROM, (uint8_t)addr);
}

static bool eeprom_read(uint16_t addr, uint8_t *v) {
    bool ok = eeprom_addr(addr);
    if (ok) {
        i2c_start(&I2C_EEPROM);
        ok = i2c_write_byte(&I2C_EEPROM, (EEPROM_ADDR << 1) | 1u);
        if (ok) *v = i2c_read_byte(&I2C_EEPROM, false);
    }
    i2c_stop(&I2C_EEPROM);
    return ok;
}

static bool eeprom_write(uint16_t addr, uint8_t v) {
    bool ok = eeprom_addr(addr) && i2c_write_byte(&I2C_EEPROM, v);
    i2c_stop(&I2C_EEPROM);
    uint32_t start = now_ms();  // ACK polling through the 5 ms write cycle
    while (ok && !i2c_probe(&I2C_EEPROM, EEPROM_ADDR)) {
        if ((uint32_t)(now_ms() - start) > 20u) return false;
    }
    return ok;
}

static void test_eeprom_rw(void) {
    const uint16_t addr = 0xFFFFu;
    uint8_t orig = 0, back = 0, restored = 0;
    if (!i2c_init(&I2C_EEPROM) || !eeprom_read(addr, &orig)) {
        fail("EEPROM write/read", "no ACK");
        return;
    }
    uint8_t pattern = (uint8_t)~orig;
    bool ok = eeprom_write(addr, pattern) && eeprom_read(addr, &back) && back == pattern;
    bool ok2 = eeprom_write(addr, orig) && eeprom_read(addr, &restored) && restored == orig;
    check(ok && ok2, "EEPROM write/read", "0x%04x: wrote 0x%02x, read 0x%02x, restored 0x%02x", addr, pattern, back,
          restored);
}

// ---- Connector and microSD pins: shorts to rails, ground or each other ----

static void ext_all(uint32_t pull) {
    for (unsigned i = 0; i < EXT_PIN_COUNT; i++) pin_input(EXT_PINS[i].pin, pull);
    delay_us(50);
}

static void test_ext_pins(void) {
    int bad = 0;
    // Each pin must follow the MCU's 40 kOhm pull both ways, unless the board pulls it.
    ext_all(PullUp);
    bool up[EXT_PIN_COUNT];
    for (unsigned i = 0; i < EXT_PIN_COUNT; i++) up[i] = pin_read(EXT_PINS[i].pin);
    ext_all(PullDown);
    for (unsigned i = 0; i < EXT_PIN_COUNT; i++) {
        const ExtPin *e = &EXT_PINS[i];
        bool down = pin_read(e->pin);
        bool want_up = e->ext != ExtDown, want_down = e->ext == ExtUp;
        if (up[i] != want_up || down != want_down) {
            fail(e->name, "reads %s with pull-up, %s with pull-down (expect %s, %s)", up[i] ? "high" : "low",
                 down ? "high" : "low", want_up ? "high" : "low", want_down ? "high" : "low");
            bad++;
        }
    }
    // Drive one pin against the others' pulls; any that follows is shorted to it.
    for (int phase = 0; phase < 2; phase++) {
        bool drive_high = phase == 0;
        ext_all(drive_high ? PullDown : PullUp);
        for (unsigned i = 0; i < EXT_PIN_COUNT; i++) {
            pin_output(EXT_PINS[i].pin, drive_high, false);
            delay_us(20);
            for (unsigned j = 0; j < EXT_PIN_COUNT; j++) {
                const ExtPin *e = &EXT_PINS[j];
                if (j == i || e->ext == (drive_high ? ExtUp : ExtDown)) continue;  // already at that level
                if (pin_read(e->pin) == drive_high) {
                    fail("Pin short", "%s follows %s driven %s", e->name, EXT_PINS[i].name, drive_high ? "high" : "low");
                    bad++;
                }
            }
            pin_input(EXT_PINS[i].pin, drive_high ? PullDown : PullUp);
            delay_us(20);
        }
    }
    ext_all(PullNone);
    if (!bad) pass("Connector and microSD pins", "%u pins: no shorts to a rail, ground or each other", (uint32_t)EXT_PIN_COUNT);
}

// ---- Carrier translators: with no carrier, U8/U10 still drive their STM32-side outputs ----

static void test_carrier_translators(void) {
    int driven = 0;
    for (unsigned i = 0; i < sizeof CARRIER_IN / sizeof CARRIER_IN[0]; i++) {
        pin_input(CARRIER_IN[i], PullUp);
        delay_us(50);
        bool hi = pin_read(CARRIER_IN[i]);
        pin_input(CARRIER_IN[i], PullDown);
        delay_us(50);
        bool lo = pin_read(CARRIER_IN[i]);
        pin_input(CARRIER_IN[i], PullNone);
        if (hi == lo) driven++;
        info(CARRIER_IN_NAME[i], hi == lo ? "driven %s by the translator" : "follows the MCU pull (not driven)",
             hi ? "high" : "low");
    }
    check(driven == 4, "Carrier translators U8/U10", "%d of 4 outputs driven (needs P1V8 and OE)", driven);
}

// ---- LEDs: one at a time, for the eye ----

static void test_leds(void) {
    con_puts("[INFO] LEDs: red, green, blue for 0.4 s each\n");
    leds(true, false, false);
    delay_ms(400);
    leds(false, true, false);
    delay_ms(400);
    leds(false, false, true);
    delay_ms(400);
}

// ---- ExpressLRS ESP32-C3: reset it and read its ROM boot line on UART4 ----

#define ELRS_UART UART4

// Captures UART4 for `window_ms` after releasing CHIP_EN and returns the
// "rst:...,boot:..." line the ESP32-C3 ROM prints at 115200 (esptool docs).
static bool elrs_boot_line(bool download, char *line, int cap, uint32_t window_ms) {
    static char buf[2048];
    int n = 0;
    pin_output(ELRS_BOOT, !download, true);  // open drain; receiver pull-ups
    pin_output(ELRS_EN, false, true);
    delay_ms(20);  // 1 kOhm into 1 uF: CHIP_EN well below threshold
    uart_drain(ELRS_UART);
    pin_write(ELRS_EN, true);  // 10 kOhm / 1 uF releases it about 10 ms later
    uint32_t start = now_ms();
    while ((uint32_t)(now_ms() - start) < window_ms && n < (int)sizeof(buf) - 1) {
        int c = uart_getc(ELRS_UART);
        if (c >= 0) buf[n++] = (char)c;
    }
    buf[n] = 0;
    pin_write(ELRS_BOOT, true);  // strap is latched; release BOOT
    for (int i = 0; i + 4 < n; i++) {
        if (buf[i] == 'r' && buf[i + 1] == 's' && buf[i + 2] == 't' && buf[i + 3] == ':') {
            int k = 0;
            while (i + k < n && buf[i + k] != '\r' && buf[i + k] != '\n' && k < cap - 1) {
                line[k] = buf[i + k];
                k++;
            }
            line[k] = 0;
            return true;
        }
    }
    line[0] = 0;
    return false;
}

static bool contains(const char *s, const char *needle) {
    for (; *s; s++) {
        const char *a = s, *b = needle;
        while (*a && *b && *a == *b) {
            a++;
            b++;
        }
        if (!*b) return true;
    }
    return false;
}

static void test_elrs(void) {
    uart_init(ELRS_UART, ELRS_TX, ELRS_RX, ELRS_AF, 115200);
    char line[96];

    bool got = elrs_boot_line(true, line, sizeof(line), 500);
    check(got && contains(line, "DOWNLOAD"), "ELRS download strap (PG5)", got ? "%s" : "no ROM boot line on UART4", line);

    got = elrs_boot_line(false, line, sizeof(line), 500);
    check(got && contains(line, "SPI_FAST_FLASH_BOOT"), "ELRS reset + UART4 (PG6)",
          got ? "%s" : "no ROM boot line on UART4 (check PG6, UART4, receiver 3.3 V)", line);

    pin_input(ELRS_BOOT, PullNone);  // leave both released to the receiver pull-ups
    pin_input(ELRS_EN, PullNone);
}

// ---- ExpressLRS passthrough: UART6 (probe's second console) <-> UART4 ----
//
// 'p' holds BOOT low and resets the ESP32-C3 into its ROM loader at 115200 for
// esptool (--before no_reset --after no_reset); 'c' resets it into its
// firmware and bridges at the 420000 baud CRSF rate. Either runs until reset.

static void elrs_bridge(bool download) {
    uint32_t baud = download ? 115200u : 420000u;
    con_printf("[INFO] ELRS passthrough: %s, UART6 <-> UART4 at %u baud until reset\n",
               download ? "ROM download mode" : "normal boot", baud);
    RCC->APB2ENR |= RCC_APB2ENR_USART6EN;
    (void)RCC->APB2ENR;
    uart_init(USART6, UART6_TX, UART6_RX, UART6_AF, baud);
    uart_init(ELRS_UART, ELRS_TX, ELRS_RX, ELRS_AF, baud);
    pin_output(ELRS_BOOT, !download, true);
    pin_output(ELRS_EN, false, true);
    delay_ms(20);
    pin_write(ELRS_EN, true);
    // BOOT stays low in download mode: a blank part resets itself (RTC watchdog
    // after "invalid header") and must land back in the loader every time.
    delay_ms(100);  // EN releases about 10 ms later; the ROM samples BOOT then
    // 16-byte hardware FIFOs (FIFOEN only while UE = 0) and a ring buffer per
    // direction, so neither side ever waits on the other.
    USART_TypeDef *const ports[2] = { USART6, ELRS_UART };
    for (int i = 0; i < 2; i++) {
        ports[i]->CR1 &= ~USART_CR1_UE;
        ports[i]->CR1 |= USART_CR1_FIFOEN | USART_CR1_UE;
    }
    static uint8_t ring[2][1024];
    uint32_t head[2] = { 0, 0 }, tail[2] = { 0, 0 };  // ring[i] carries ports[i] -> ports[1 - i]
    leds(false, false, true);
    while (1) {
        for (int i = 0; i < 2; i++) {
            USART_TypeDef *in = ports[i], *out = ports[1 - i];
            if ((in->ISR & (USART_ISR_ORE | USART_ISR_FE | USART_ISR_NE)) != 0u) {
                in->ICR = USART_ICR_ORECF | USART_ICR_FECF | USART_ICR_NECF;
                GPIOE->BSRR = 1u << LED_RED.pin;  // red: a byte was lost
            }
            while ((in->ISR & USART_ISR_RXNE_RXFNE) != 0u && head[i] - tail[i] < sizeof ring[i]) {
                ring[i][head[i]++ % sizeof ring[i]] = (uint8_t)in->RDR;
            }
            while (tail[i] != head[i] && (out->ISR & USART_ISR_TXE_TXFNF) != 0u) {
                out->TDR = ring[i][tail[i]++ % sizeof ring[i]];
            }
        }
    }
}

// ---- Run ----

static void run(void) {
    passes = fails = warnings = 0;
    leds(false, false, true);
    con_puts("\n=== Aleph Flight Controller self-test ===\n");
    test_identity();
    test_reset_cause();
    test_supply();
    test_clocks();
    test_static_levels();
    test_i2c();
    test_fram();
    test_spi_imu();
    test_ins();
    test_adc();
    test_elrs();
    test_can();
    test_uart6();
    test_eeprom_rw();
    test_ext_pins();
    test_carrier_translators();
    test_leds();
    con_printf("=== %u passed, %u failed, %u warnings ===\n", passes, fails, warnings);
    con_puts("Not covered: USB, CAN bus traffic, microSD cards, P5V_AUX voltage, connector opens.\n");
    con_puts("Press 'r' to run again, 'p' for ELRS flashing passthrough, 'c' for ELRS CRSF passthrough,\n");
    con_puts("'i' for the IMX-6 deep test, 's' for live IMX-6 data, 'x' for IMX-6 SET_DATA checks. ('g' tries to enable G9 data ready: crashes firmware 3.1.1.0)\n");
    leds(fails != 0u, fails == 0u, false);
}

int main(void) {
    // Already in LDO mode from Reset_Handler. Reset flags first, then clear them.
    reset_flags = RCC->RSR;
    RCC->RSR |= RCC_RSR_RMVF;

    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOAEN | RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOCEN | RCC_AHB4ENR_GPIODEN |
                    RCC_AHB4ENR_GPIOEEN | RCC_AHB4ENR_GPIOFEN | RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN |
                    RCC_AHB4ENR_GPIOIEN | RCC_AHB4ENR_GPIOJEN | RCC_AHB4ENR_GPIOKEN;
    RCC->APB1LENR |= RCC_APB1LENR_USART3EN | RCC_APB1LENR_UART4EN;
    (void)RCC->APB1LENR;

    time_init();
    pin_output(LED_RED, false, false);
    pin_output(LED_GREEN, false, false);
    pin_output(LED_BLUE, false, false);
    uart_init(CONSOLE, CONSOLE_TX, CONSOLE_RX, CONSOLE_AF, 115200);

    // P3V3_SENS1 starts up to 40 ms after P3V3 (spec): let it settle.
    delay_ms(100);
    run();

    uint32_t blink = now_ms();
    while (1) {
        int key = uart_getc(CONSOLE);
        if (key == 'r') run();
        if (key == 'p' || key == 'c') elrs_bridge(key == 'p');
        if (key == 'i') ins_deep();
        if (key == 'g') ins_enable_drdy();
        if (key == 'x') ins_set_tests();
        if (key == 's') ins_live();
        if ((uint32_t)(now_ms() - blink) >= 500u) {  // heartbeat on the result LED
            blink = now_ms();
            if (fails) GPIOE->ODR ^= 1u << LED_RED.pin;
            else GPIOA->ODR ^= 1u << LED_GREEN.pin;
        }
    }
}

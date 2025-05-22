use crate::i2c_dma;
use core::ops::DerefMut;
use embedded_hal::i2c::I2c as I2cTrait;
use hal::i2c;

const BASE_ADDRESS: u8 = 0x50; // Base address for FM24CL16B
const TOTAL_SIZE: usize = 2048; // 2KB total (16Kbit)
const PAGE_SIZE: usize = 256; // 256 bytes per page

#[derive(Debug, defmt::Format)]
pub enum Error {
    I2cDma(i2c_dma::Error),
    CommunicationFailed,
    AddressOutOfRange,
    InvalidLength,
}

impl From<i2c::Error> for Error {
    fn from(err: i2c::Error) -> Self {
        Error::I2cDma(i2c_dma::Error::I2c(err))
    }
}

impl From<i2c_dma::Error> for Error {
    fn from(err: i2c_dma::Error) -> Self {
        Error::I2cDma(err)
    }
}

pub struct Fm24cl16b {
    i2c_address: u8,
    current_page: u8,
}

impl Fm24cl16b {
    pub fn new(i2c_dma: &mut i2c_dma::I2cDma) -> Result<Self, Error> {
        let fram = Self {
            i2c_address: BASE_ADDRESS,
            current_page: 0,
        };

        // Validate communication by directly using embedded_hal I2C trait
        defmt::debug!(
            "Validating FM24CL16B communication at address 0x{:02x}",
            fram.i2c_address
        );
        match I2cTrait::write(i2c_dma.deref_mut(), fram.i2c_address, &[]) {
            Ok(_) => {
                defmt::info!("FM24CL16B initialized successfully");
                Ok(fram)
            }
            Err(e) => {
                defmt::error!("FM24CL16B initialization failed: {:?}", e);
                Err(Error::I2cDma(i2c_dma::Error::I2c(e)))
            }
        }
    }

    fn set_page(&mut self, page: u8) {
        let page = page & 0x07; // Ensure page is in range 0-7
        self.current_page = page;
        // The FM24CL16B uses the upper 3 bits of the I2C address as page select
        // Each page is at a different I2C address: 0x50 + page
        self.i2c_address = BASE_ADDRESS | page;
    }

    fn calculate_address(&self, address: usize) -> Result<(u8, u8), Error> {
        if address >= TOTAL_SIZE {
            return Err(Error::AddressOutOfRange);
        }

        let page = (address / PAGE_SIZE) as u8;
        let word_address = (address % PAGE_SIZE) as u8;
        Ok((page, word_address))
    }

    pub fn read_byte(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        address: usize,
    ) -> Result<u8, Error> {
        let (page, word_address) = self.calculate_address(address)?;

        // Set correct page if needed
        if page != self.current_page {
            self.set_page(page);
        }

        // Set word address then read byte using direct I2C calls
        I2cTrait::write(i2c_dma.deref_mut(), self.i2c_address, &[word_address])?;
        let mut data = [0u8];
        I2cTrait::read(i2c_dma.deref_mut(), self.i2c_address, &mut data)?;

        Ok(data[0])
    }

    pub fn write_byte(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        address: usize,
        value: u8,
    ) -> Result<(), Error> {
        let (page, word_address) = self.calculate_address(address)?;

        if page != self.current_page {
            self.set_page(page);
        }

        // Write using direct I2C call
        I2cTrait::write(
            i2c_dma.deref_mut(),
            self.i2c_address,
            &[word_address, value],
        )?;

        Ok(())
    }

    pub fn read(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        start_address: usize,
        buffer: &mut [u8],
    ) -> Result<(), Error> {
        if start_address + buffer.len() > TOTAL_SIZE {
            return Err(Error::AddressOutOfRange);
        }

        let mut address = start_address;
        let mut buffer_offset = 0;

        while buffer_offset < buffer.len() {
            let (page, word_address) = self.calculate_address(address)?;

            if page != self.current_page {
                self.set_page(page);
            }

            // Calculate how many bytes to read from current page
            let bytes_left_in_page = PAGE_SIZE - (word_address as usize);
            let bytes_to_read = core::cmp::min(bytes_left_in_page, buffer.len() - buffer_offset);

            // Set word address and read data using direct I2C calls
            I2cTrait::write(i2c_dma.deref_mut(), self.i2c_address, &[word_address])?;
            I2cTrait::read(
                i2c_dma.deref_mut(),
                self.i2c_address,
                &mut buffer[buffer_offset..(buffer_offset + bytes_to_read)],
            )?;

            address += bytes_to_read;
            buffer_offset += bytes_to_read;
        }

        Ok(())
    }

    pub fn write(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        start_address: usize,
        data: &[u8],
    ) -> Result<(), Error> {
        if start_address + data.len() > TOTAL_SIZE {
            return Err(Error::AddressOutOfRange);
        }

        let mut address = start_address;
        let mut data_offset = 0;

        while data_offset < data.len() {
            let (page, word_address) = self.calculate_address(address)?;

            if page != self.current_page {
                self.set_page(page);
            }

            // Calculate bytes to write within current page boundary
            let bytes_left_in_page = PAGE_SIZE - (word_address as usize);
            let bytes_to_write = core::cmp::min(bytes_left_in_page, data.len() - data_offset);

            // Limit to 32 bytes per write (common I2C buffer limitation)
            let safe_bytes_to_write = core::cmp::min(bytes_to_write, 32);

            // Prepare write buffer: word address + data
            let mut write_buffer = [0u8; 33]; // 1 byte word address + max 32 bytes data
            write_buffer[0] = word_address;
            write_buffer[1..(safe_bytes_to_write + 1)]
                .copy_from_slice(&data[data_offset..(data_offset + safe_bytes_to_write)]);

            // Write using direct I2C call
            I2cTrait::write(
                i2c_dma.deref_mut(),
                self.i2c_address,
                &write_buffer[0..(safe_bytes_to_write + 1)],
            )?;

            address += safe_bytes_to_write;
            data_offset += safe_bytes_to_write;
        }

        Ok(())
    }

    pub fn size(&self) -> usize {
        TOTAL_SIZE
    }

    pub fn self_test(&mut self, i2c: &mut i2c_dma::I2cDma) {
        defmt::info!("Running FRAM self-test");

        // Test pattern at the beginning of memory
        let test_patterns = [
            [0xAA, 0x55, 0xFF, 0x00], // Mixed pattern
            [0x12, 0x34, 0x56, 0x78], // Ascending values
        ];

        // Test at beginning of memory
        for (i, pattern) in test_patterns.iter().enumerate() {
            let addr = i * 8; // Space them out

            // Write pattern
            match self.write(i2c, addr, pattern) {
                Ok(_) => {}
                Err(e) => panic!("FRAM write failed at addr {}: {:?}", addr, e),
            }

            // Read back and verify
            let mut read_data = [0u8; 4];
            match self.read(i2c, addr, &mut read_data) {
                Ok(_) => {}
                Err(e) => panic!("FRAM read failed at addr {}: {:?}", addr, e),
            }

            // Verify match
            for (j, &byte) in pattern.iter().enumerate() {
                if byte != read_data[j] {
                    panic!(
                        "Pattern {} failed at byte {}: expected 0x{:02x}, got 0x{:02x}",
                        i, j, byte, read_data[j]
                    );
                }
            }
        }

        // Test a page boundary (256 bytes)
        let page_boundary = 256;
        let cross_page_addr = page_boundary - 2;
        let cross_page_data = [0xAB, 0xCD, 0xEF, 0x12]; // crosses from page 0 to page 1

        match self.write(i2c, cross_page_addr, &cross_page_data) {
            Ok(_) => {}
            Err(e) => panic!("FRAM write failed at page boundary: {:?}", e),
        }

        let mut read_cross_page = [0u8; 4];
        match self.read(i2c, cross_page_addr, &mut read_cross_page) {
            Ok(_) => {}
            Err(e) => panic!("FRAM read failed at page boundary: {:?}", e),
        }

        for (i, &byte) in cross_page_data.iter().enumerate() {
            if byte != read_cross_page[i] {
                panic!(
                    "Page boundary test failed at byte {}: expected 0x{:02x}, got 0x{:02x}",
                    i, byte, read_cross_page[i]
                );
            }
        }

        // Test end of memory
        let end_addr = self.size() - 4;
        let end_data = [0xDE, 0xAD, 0xBE, 0xEF];

        match self.write(i2c, end_addr, &end_data) {
            Ok(_) => {}
            Err(e) => panic!("FRAM write failed at end address {}: {:?}", end_addr, e),
        }

        let mut read_end_data = [0u8; 4];
        match self.read(i2c, end_addr, &mut read_end_data) {
            Ok(_) => {}
            Err(e) => panic!("FRAM read failed at end address {}: {:?}", end_addr, e),
        }

        for (i, &byte) in end_data.iter().enumerate() {
            if byte != read_end_data[i] {
                panic!(
                    "End memory test failed at byte {}: expected 0x{:02x}, got 0x{:02x}",
                    i, byte, read_end_data[i]
                );
            }
        }

        defmt::info!("All FRAM self-tests passed successfully!");
    }
}

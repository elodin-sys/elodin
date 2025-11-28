/// OSD Grid - In-memory text grid representation for OSD
pub struct OsdGrid {
    pub rows: u8,
    pub cols: u8,
    cells: Vec<char>,
}

impl OsdGrid {
    pub fn new(rows: u8, cols: u8) -> Self {
        let len = rows as usize * cols as usize;
        Self {
            rows,
            cols,
            cells: vec![' '; len],
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.cells {
            *c = ' ';
        }
    }

    /// Write ASCII/UTF-8 text at (row, col), clipped to the grid.
    pub fn write_text(&mut self, row: u8, col: u8, text: &str) {
        let row = row as usize;
        let mut col = col as usize;
        let cols = self.cols as usize;

        if row >= self.rows as usize || col >= cols {
            return;
        }

        for ch in text.chars() {
            if col >= cols {
                break;
            }
            let idx = row * cols + col;
            self.cells[idx] = ch;
            col += 1;
        }
    }

    /// Write text centered on a row
    pub fn write_centered(&mut self, row: u8, text: &str) {
        let text_len = text.chars().count();
        let cols = self.cols as usize;
        if text_len >= cols {
            self.write_text(row, 0, text);
        } else {
            let start_col = (cols - text_len) / 2;
            self.write_text(row, start_col as u8, text);
        }
    }

    /// Write text right-aligned on a row
    #[allow(dead_code)]
    pub fn write_right_aligned(&mut self, row: u8, text: &str) {
        let text_len = text.chars().count();
        let cols = self.cols as usize;
        if text_len >= cols {
            self.write_text(row, 0, text);
        } else {
            let start_col = cols - text_len;
            self.write_text(row, start_col as u8, text);
        }
    }

    /// Draw a vertical line
    #[allow(dead_code)]
    pub fn draw_vertical_line(&mut self, col: u8, start_row: u8, end_row: u8, ch: char) {
        for row in start_row..=end_row.min(self.rows - 1) {
            self.set_char(row, col, ch);
        }
    }

    /// Draw a horizontal line
    #[allow(dead_code)]
    pub fn draw_horizontal_line(&mut self, row: u8, start_col: u8, end_col: u8, ch: char) {
        for col in start_col..=end_col.min(self.cols - 1) {
            self.set_char(row, col, ch);
        }
    }

    /// Set a single character at (row, col)
    pub fn set_char(&mut self, row: u8, col: u8, ch: char) {
        let row = row as usize;
        let col = col as usize;
        let cols = self.cols as usize;

        if row < self.rows as usize && col < cols {
            let idx = row * cols + col;
            self.cells[idx] = ch;
        }
    }

    /// Get a character at (row, col)
    #[allow(dead_code)]
    pub fn get_char(&self, row: u8, col: u8) -> Option<char> {
        let row = row as usize;
        let col = col as usize;
        let cols = self.cols as usize;

        if row < self.rows as usize && col < cols {
            let idx = row * cols + col;
            Some(self.cells[idx])
        } else {
            None
        }
    }

    /// Return a given row as a String (for backends to render/encode).
    pub fn line_as_str(&self, row: u8) -> String {
        let row = row as usize;
        let cols = self.cols as usize;
        if row >= self.rows as usize {
            return String::new();
        }
        let start = row * cols;
        let end = start + cols;
        self.cells[start..end].iter().collect()
    }

    /// Get all non-empty lines with their row indices
    pub fn non_empty_lines(&self) -> Vec<(u8, String)> {
        let mut lines = Vec::new();
        for row in 0..self.rows {
            let line = self.line_as_str(row);
            let trimmed = line.trim_end();
            if !trimmed.is_empty() {
                lines.push((row, trimmed.to_string()));
            }
        }
        lines
    }

    /// Draw a box around a region
    #[allow(dead_code)]
    pub fn draw_box(&mut self, top: u8, left: u8, bottom: u8, right: u8) {
        // Corners
        self.set_char(top, left, '┌');
        self.set_char(top, right, '┐');
        self.set_char(bottom, left, '└');
        self.set_char(bottom, right, '┘');

        // Top and bottom borders
        for col in (left + 1)..right {
            self.set_char(top, col, '─');
            self.set_char(bottom, col, '─');
        }

        // Left and right borders
        for row in (top + 1)..bottom {
            self.set_char(row, left, '│');
            self.set_char(row, right, '│');
        }
    }
}

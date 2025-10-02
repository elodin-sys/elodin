# How-to add a new "EQL formula"
When extending EQL with a new formula, there are two possible scenarios:

## Scenario A
The mirror-function already exists in PostgreSQL. You can directly map it in EQL.

Suppose you want to add a `norm()` formula that computes the vector norm. Since PostgreSQL already provides the mathematical function `sqrt` by default, you don’t need to implement anything at the database level. 

Instead, you just declare the new formula in the EQL layer, here [libs/db/eql/src/lib.rs](libs/db/eql/src/lib.rs).

## Scenario B

When the new formula cannot be expressed with an existing SQL primitive, you must implement it yourself and register it as a `User Defined Function` (UDF) in `DataFusion`.

For example The `fft` function does not exist in PostgreSQL and requires a custom Rust implementation.

### Step 1 – Implement the function

Add the Rust implementation in: [libs/db/src/arrow/fft.rs](libs/db/src/arrow/fft.rs). 

```rust
pub struct FftUDF {}

impl FftUDF {
    pub fn new() -> Self {
        FftUDF {}
    }
}

// Example skeleton
impl ScalarUDFImpl for FftUDF {
    fn name(&self) -> &str {
        "fft"
    }

    fn invoke(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
    }
}
```

### Step 2 – Register the function

Once implemented, you must register the UDF in: [libs/db/src/arrow/mod.rs](libs/db/src/arrow/mod.rs).  

```rust
ctx.register_udf(datafusion::logical_expr::ScalarUDF::new_from_impl(
    FftUDF::new(),
));
```

### Step 3 - Declare the new formula in the EQL layer

Exactly like the unique step on the scenario A. 
Declare the new formula in the EQL layer, here [libs/db/eql/src/lib.rs](libs/db/eql/src/lib.rs).
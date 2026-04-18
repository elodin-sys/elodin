use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    F64,
    F32,
    I1,
    I32,
    I64,
    UI32,
    UI64,
}

impl ElementType {
    pub fn byte_size(self) -> usize {
        match self {
            ElementType::F64 => 8,
            ElementType::F32 => 4,
            ElementType::I1 => 1,
            ElementType::I32 => 4,
            ElementType::I64 => 8,
            ElementType::UI32 => 4,
            ElementType::UI64 => 8,
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElementType::F64 => write!(f, "f64"),
            ElementType::F32 => write!(f, "f32"),
            ElementType::I1 => write!(f, "i1"),
            ElementType::I32 => write!(f, "i32"),
            ElementType::I64 => write!(f, "i64"),
            ElementType::UI32 => write!(f, "ui32"),
            ElementType::UI64 => write!(f, "ui64"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub shape: Vec<i64>,
    pub element_type: ElementType,
}

impl TensorType {
    pub fn scalar(element_type: ElementType) -> Self {
        Self {
            shape: vec![],
            element_type,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<i64>().max(1) as usize
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.element_type.byte_size()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor<")?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, "x")?;
            }
            write!(f, "{d}")?;
        }
        if !self.shape.is_empty() {
            write!(f, "x")?;
        }
        write!(f, "{}>", self.element_type)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareDirection {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareType {
    Float,
    Signed,
    Unsigned,
    TotalOrder,
}

#[derive(Debug, Clone)]
pub enum ScalarValue {
    F64(f64),
    F32(f32),
    I1(bool),
    I32(i32),
    I64(i64),
    UI32(u32),
    UI64(u64),
}

impl ScalarValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            ScalarValue::F64(v) => *v,
            ScalarValue::F32(v) => *v as f64,
            ScalarValue::I64(v) => *v as f64,
            ScalarValue::I32(v) => *v as f64,
            ScalarValue::UI32(v) => *v as f64,
            ScalarValue::UI64(v) => *v as f64,
            ScalarValue::I1(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            ScalarValue::I64(v) => *v,
            ScalarValue::I32(v) => *v as i64,
            ScalarValue::UI32(v) => *v as i64,
            ScalarValue::UI64(v) => *v as i64,
            ScalarValue::F64(v) => *v as i64,
            ScalarValue::F32(v) => *v as i64,
            ScalarValue::I1(v) => i64::from(*v),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConstantValue {
    DenseScalar(ScalarValue),
    DenseArray(Vec<ScalarValue>),
    DenseSplat(ScalarValue, TensorType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Add,
    Minimum,
    Maximum,
    And,
    Or,
}

#[derive(Debug, Clone)]
pub struct DotDims {
    pub lhs_contracting: Vec<i64>,
    pub rhs_contracting: Vec<i64>,
    pub lhs_batch: Vec<i64>,
    pub rhs_batch: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct GatherDims {
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
    pub index_vector_dim: i64,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Constant {
        value: ConstantValue,
    },
    Add {
        lhs: ValueId,
        rhs: ValueId,
    },
    Subtract {
        lhs: ValueId,
        rhs: ValueId,
    },
    Multiply {
        lhs: ValueId,
        rhs: ValueId,
    },
    Divide {
        lhs: ValueId,
        rhs: ValueId,
    },
    Negate {
        operand: ValueId,
    },
    Sqrt {
        operand: ValueId,
    },
    Maximum {
        lhs: ValueId,
        rhs: ValueId,
    },
    Compare {
        lhs: ValueId,
        rhs: ValueId,
        direction: CompareDirection,
        compare_type: CompareType,
    },
    Select {
        cond: ValueId,
        on_true: ValueId,
        on_false: ValueId,
    },
    Reshape {
        operand: ValueId,
    },
    BroadcastInDim {
        operand: ValueId,
        broadcast_dims: Vec<i64>,
    },
    Slice {
        operand: ValueId,
        start_indices: Vec<i64>,
        limit_indices: Vec<i64>,
    },
    Concatenate {
        operands: Vec<ValueId>,
        dimension: i64,
    },
    DotGeneral {
        lhs: ValueId,
        rhs: ValueId,
        dims: DotDims,
    },
    Reduce {
        operand: ValueId,
        init: ValueId,
        op: ReduceOp,
        dimensions: Vec<i64>,
    },
    ReduceArgminmax {
        values: ValueId,
        indices: ValueId,
        dimensions: Vec<i64>,
        is_min: bool,
    },
    Convert {
        operand: ValueId,
    },
    BitcastConvert {
        operand: ValueId,
    },
    Iota {
        dimension: i64,
    },
    Xor {
        lhs: ValueId,
        rhs: ValueId,
    },
    Or {
        lhs: ValueId,
        rhs: ValueId,
    },
    And {
        lhs: ValueId,
        rhs: ValueId,
    },
    ShiftLeft {
        lhs: ValueId,
        rhs: ValueId,
    },
    ShiftRightLogical {
        lhs: ValueId,
        rhs: ValueId,
    },
    Call {
        callee: String,
        args: Vec<ValueId>,
    },
    While {
        cond_body: Vec<InstrResult>,
        loop_body: Vec<InstrResult>,
        init_values: Vec<ValueId>,
        iter_arg_ids: Vec<ValueId>,
    },
    Case {
        index: ValueId,
        branches: Vec<Vec<InstrResult>>,
    },
    Return {
        operands: Vec<ValueId>,
    },
    ErfInv {
        operand: ValueId,
    },
    Gather {
        operand: ValueId,
        indices: ValueId,
        dims: GatherDims,
        slice_sizes: Vec<i64>,
    },
    Transpose {
        operand: ValueId,
        permutation: Vec<i64>,
    },
    DynamicSlice {
        operand: ValueId,
        start_indices: Vec<ValueId>,
        slice_sizes: Vec<i64>,
    },
    DynamicUpdateSlice {
        operand: ValueId,
        update: ValueId,
        start_indices: Vec<ValueId>,
    },
    Sine {
        operand: ValueId,
    },
    Cosine {
        operand: ValueId,
    },
    Atan2 {
        lhs: ValueId,
        rhs: ValueId,
    },
    Abs {
        operand: ValueId,
    },
    Minimum {
        lhs: ValueId,
        rhs: ValueId,
    },
    Sign {
        operand: ValueId,
    },
    Remainder {
        lhs: ValueId,
        rhs: ValueId,
    },
    Acos {
        operand: ValueId,
    },
    Exponential {
        operand: ValueId,
    },
    Log {
        operand: ValueId,
    },
    Clamp {
        operand: ValueId,
        min: ValueId,
        max: ValueId,
    },
    Power {
        lhs: ValueId,
        rhs: ValueId,
    },
    Reverse {
        operand: ValueId,
        dimensions: Vec<i64>,
    },
    Tanh {
        operand: ValueId,
    },
    Tan {
        operand: ValueId,
    },
    Floor {
        operand: ValueId,
    },
    RoundNearestEven {
        operand: ValueId,
    },
    Pad {
        operand: ValueId,
        padding_value: ValueId,
        low: Vec<i64>,
        high: Vec<i64>,
        interior: Vec<i64>,
    },
    Scatter {
        operand: ValueId,
        indices: ValueId,
        updates: ValueId,
    },
    CustomCall {
        call_target: String,
        operands: Vec<ValueId>,
        backend_config: HashMap<String, i64>,
    },
    Rsqrt {
        operand: ValueId,
    },
    Log1p {
        operand: ValueId,
    },
    IsFinite {
        operand: ValueId,
    },
    Not {
        operand: ValueId,
    },
    Ceil {
        operand: ValueId,
    },
    ShiftRightArithmetic {
        lhs: ValueId,
        rhs: ValueId,
    },
    Asin {
        operand: ValueId,
    },
    Atan {
        operand: ValueId,
    },
    Sinh {
        operand: ValueId,
    },
    Cosh {
        operand: ValueId,
    },
    Erfc {
        operand: ValueId,
    },
    Expm1 {
        operand: ValueId,
    },
    Cbrt {
        operand: ValueId,
    },
    Sort {
        inputs: Vec<ValueId>,
        dimension: i64,
        is_stable: bool,
        comparator: Vec<InstrResult>,
        comparator_params: Vec<ValueId>,
    },
    BatchNormInference {
        operand: ValueId,
        scale: ValueId,
        offset: ValueId,
        mean: ValueId,
        variance: ValueId,
        epsilon: f64,
        feature_index: i64,
    },
    RealDynamicSlice {
        operand: ValueId,
        start_indices: ValueId,
        limit_indices: ValueId,
        strides: ValueId,
    },
    Map {
        inputs: Vec<ValueId>,
        dimensions: Vec<i64>,
        body: Vec<InstrResult>,
        body_params: Vec<ValueId>,
    },
    ReduceWindow {
        operands: Vec<ValueId>,
        init_values: Vec<ValueId>,
        body: Vec<InstrResult>,
        body_params: Vec<ValueId>,
        window_dimensions: Vec<i64>,
        window_strides: Vec<i64>,
        base_dilations: Vec<i64>,
        window_dilations: Vec<i64>,
        padding: Vec<(i64, i64)>,
    },
    SelectAndScatter {
        operand: ValueId,
        source: ValueId,
        init_value: ValueId,
        select_body: Vec<InstrResult>,
        select_params: Vec<ValueId>,
        scatter_body: Vec<InstrResult>,
        scatter_params: Vec<ValueId>,
        window_dimensions: Vec<i64>,
        window_strides: Vec<i64>,
        padding: Vec<(i64, i64)>,
    },
    Convolution {
        lhs: ValueId,
        rhs: ValueId,
        dimension_numbers: ConvDimensionNumbers,
        window_strides: Vec<i64>,
        padding: Vec<(i64, i64)>,
        lhs_dilation: Vec<i64>,
        rhs_dilation: Vec<i64>,
        feature_group_count: i64,
        batch_group_count: i64,
    },
    CholeskyOp {
        operand: ValueId,
        lower: bool,
    },
    TriangularSolve {
        a: ValueId,
        b: ValueId,
        left_side: bool,
        lower: bool,
        unit_diagonal: bool,
        transpose_a: bool,
    },
    Fft {
        operand: ValueId,
        fft_type: FftType,
        fft_length: Vec<i64>,
    },
    Rng {
        operands: Vec<ValueId>,
        rng_distribution: RngDistribution,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftType {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngDistribution {
    Uniform,
    Normal,
}

#[derive(Debug, Clone)]
pub struct ConvDimensionNumbers {
    pub input_batch_dimension: i64,
    pub input_feature_dimension: i64,
    pub input_spatial_dimensions: Vec<i64>,
    pub kernel_input_feature_dimension: i64,
    pub kernel_output_feature_dimension: i64,
    pub kernel_spatial_dimensions: Vec<i64>,
    pub output_batch_dimension: i64,
    pub output_feature_dimension: i64,
    pub output_spatial_dimensions: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct InstrResult {
    pub values: Vec<(ValueId, TensorType)>,
    pub instr: Instruction,
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: String,
    pub is_public: bool,
    pub params: Vec<(ValueId, TensorType)>,
    pub result_types: Vec<TensorType>,
    pub body: Vec<InstrResult>,
    /// StableHLO source line of the `func.func` declaration when the
    /// parser captured it. Used by the profiler to attach source
    /// locations to Tracy zones. `None` when the parser didn't track
    /// lines for this function.
    pub source_line: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub functions: Vec<FuncDef>,
    pub func_index: HashMap<String, usize>,
}

impl Module {
    pub fn new(functions: Vec<FuncDef>) -> Self {
        let func_index = functions
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        Self {
            functions,
            func_index,
        }
    }

    pub fn get_func(&self, name: &str) -> Option<&FuncDef> {
        self.func_index.get(name).map(|&i| &self.functions[i])
    }

    pub fn main_func(&self) -> Option<&FuncDef> {
        self.get_func("main")
    }
}

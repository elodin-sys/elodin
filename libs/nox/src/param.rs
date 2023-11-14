pub struct Op;

pub struct Literal;

pub struct Buffer;

pub trait Param {
    type Inner;
}

impl Param for Op {
    type Inner = xla::XlaOp;
}

impl Param for Literal {
    type Inner = xla::Literal;
}

impl Param for Buffer {
    type Inner = xla::PjRtBuffer;
}

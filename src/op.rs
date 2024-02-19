#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op{

}

#[derive(Clone)]
pub struct BackpropOp(Option<Op>);

impl BackpropOp {
    pub fn none() -> Self {
        Self(None)
    }
}

pub trait UnaryOpT {
    const NAME: &'static str;

    const V: Self;

    fn f64(v1: f64) -> f64;
}

pub trait BinaryOpT {
    const NAME: &'static str;

    const V: Self;

}

pub struct Reciprocal;

pub struct Abs;

pub struct Neg;

impl UnaryOpT for Reciprocal {
    const NAME: &'static str = "reciprocal";

    const V: Self = Reciprocal;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        v1.recip()
    }
}

impl UnaryOpT for Abs {
    const NAME: &'static str = "abs";

    const V: Self = Abs;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        v1.abs()
    }
}

impl UnaryOpT for Neg {
    const NAME: &'static str = "neg";

    const V: Self = Neg;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        -v1
    }
}

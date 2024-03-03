use crate::tensor::Tensor;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
pub enum Op{
    Unary(Tensor, UnaryOp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Reciprocal,
    Abs,
    Neg,
    Exp,
    Relu,
    Sqr,
    Log,
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

pub struct Exp;

pub struct Relu;

pub struct Sqr;

pub struct Log;

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


impl UnaryOpT for Exp {
    const NAME: &'static str = "exp";

    const V: Self = Exp;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        v1.exp()
    }
}

impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";

    const V: Self = Relu;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        if v1 > 0.0 {
            v1
        } else {
            0.0
        }
    }
}

impl UnaryOpT for Sqr {
    const NAME: &'static str = "sqr";

    const V: Self = Sqr;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        v1 * v1
    }
}

impl UnaryOpT for Log {
    const NAME: &'static str = "log";

    const V: Self = Log;

    #[inline(always)]
    fn f64(v1: f64) -> f64 {
        v1.ln()
    }
}

#[derive(Debug, Clone)]
pub struct BackpropOp(Option<Op>);

impl BackpropOp {

    pub fn new1(arg: &Tensor, f: impl Fn(Tensor) -> Op) -> Self {
       if arg.track_op() {
           Self(Some(f(arg.clone())))
       } else {
           Self(None)
       }
    }
    pub fn none() -> Self {
        Self(None)
    }
}

impl std::ops::Deref for BackpropOp {
    type Target = Option<Op>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


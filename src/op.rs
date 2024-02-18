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
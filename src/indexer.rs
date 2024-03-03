use std::ops::{*};
use crate::tensor::Tensor;


#[derive(Debug, Clone)]
pub enum TensorIndexer {
    Narrow(Bound<usize>, Bound<usize>),
}

pub trait RB:RangeBounds<usize> {}
impl RB for Range<usize> {}
impl RB for RangeInclusive<usize> {}
impl RB for RangeFrom<usize> {}
impl RB for RangeToInclusive<usize> {}

impl RB for RangeTo<usize> {}

impl <T: RB> From<T> for TensorIndexer {

    fn from(range: T) -> Self {
        let start = match range.start_bound() {
            Bound::Included(idx) => Bound::Included(*idx),
            Bound::Excluded(idx) => Bound::Excluded(*idx),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end = match range.end_bound() {
            Bound::Included(idx) => Bound::Included(*idx),
            Bound::Excluded(idx) => Bound::Excluded(*idx),
            Bound::Unbounded => Bound::Unbounded,
        };
        TensorIndexer::Narrow(start, end)
    }
}

pub trait IndexOp<T> {
    fn i(&self, indexers: T) -> Result<Tensor, Box<dyn std::error::Error>>;
}

impl<T> IndexOp<T> for Tensor where T: Into<TensorIndexer> {
    fn i(&self, indexers: T) -> Result<Tensor, Box<dyn std::error::Error>> {
        let indexers = vec![indexers.into()];
        self.index(&indexers)
    }
}

impl Tensor {

    fn index(&self, indexers: &[TensorIndexer]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let dims = self.shape().dims();
        let mut x = self.clone();
        let mut current_dim = 0;
        for (i, indexer) in indexers.iter().enumerate() {
            match indexer {
                TensorIndexer::Narrow(start, end) => {
                    let start = match start {
                        Bound::Included(v) => *v,
                        Bound::Excluded(v) => *v + 1,
                        Bound::Unbounded => 0,
                    };
                    let end = match end {
                        Bound::Included(v) => *v + 1,
                        Bound::Excluded(v) => *v,
                        Bound::Unbounded => dims[i],
                    };
                    let len = end - start;
                    x = x.narrow(current_dim, start, len)?;
                    current_dim += 1;
                }
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::device::Device;
    use crate::dtype::Dtype;
    use crate::shape::Shape;
    use super::*;

    #[test]
    fn test_indexer() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &device).unwrap();
        let b = a.i(0..2).unwrap();
        let b_shape = b.shape().dims();
        let b_start_offset = b.start_offset();
        assert_eq!(b_shape, &[2]);
        assert_eq!(b_start_offset, 0);
    }

    #[test]
    fn test_indexer_2() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &device).unwrap();
        let b = a.i(0..=2).unwrap();
        let b_shape = b.shape().dims();
        let b_start_offset = b.start_offset();
        assert_eq!(b_shape, &[3]);
        assert_eq!(b_start_offset, 0);
    }

    #[test]
    fn test_indexer_3() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &device).unwrap();
        let b = a.i(..=2).unwrap();
        let b_shape = b.shape().dims();
        let b_start_offset = b.start_offset();
        assert_eq!(b_shape, &[3]);
        assert_eq!(b_start_offset, 0);
    }

    #[test]
    fn test_indexer_4() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &device).unwrap();
        let b = a.i(..2).unwrap();
        let b_shape = b.shape().dims();
        let b_start_offset = b.start_offset();
        assert_eq!(b_shape, &[2]);
        assert_eq!(b_start_offset, 0);
    }

    #[test]
    fn test_indexer_5() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &device).unwrap();
        let b = a.i(1..2).unwrap();
        let b_shape = b.shape().dims();
        let b_start_offset = b.start_offset();
        assert_eq!(b_shape, &[1]);
        assert_eq!(b_start_offset, 1);
    }

    #[test]
    fn test_indexer_6() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 2]), Dtype::F64, &device).unwrap();
        let b = a.i(0..1).unwrap();
        let b_shape = b.shape().dims();
        let b_start_offset = b.start_offset();
        assert_eq!(b_shape, &[1, 2]);
        assert_eq!(b_start_offset, 0);
    }


}
use crate::shape::Shape;
use crate::strided_index::{StridedBlocks, StridedIndex};

#[derive(Debug)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    start_offset: usize,
}


impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape,
            strides: stride,
            start_offset
        }
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self::new(shape, stride, start_offset)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let dims = self.dims();
        if dim >= dims.len() {
            return Err("Dimension out of range".into());
        }

        if start + len > dims[dim] {
            return Err("Narrowing out of range".into());
        }

        let mut dims = dims.to_vec();
        dims[dim] = len;
        Ok(Self::new(Shape::from(dims), self.strides.clone(), self.start_offset + start * self.strides[dim]))
    }
    
    pub fn strided_blocks(&self) -> StridedBlocks {
        let mut block_len = 1;
        let mut contiguous_dims = 0; // These are counted from the right.
        for (&stride, &dim) in self.strides.iter().zip(self.shape.dims().iter()).rev() {
            if stride != block_len {
                break;
            }
            block_len *= dim;
            contiguous_dims += 1;
        }
        let index_dims = self.shape.dims().len() - contiguous_dims;
        if index_dims == 0 {
            StridedBlocks::SingleBlock {
                start_offset: self.start_offset,
                len: block_len,
            }
        } else {
            let block_start_index = StridedIndex::new(
                &self.dims()[..index_dims],
                &self.strides[..index_dims],
                self.start_offset,
            );
            StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            }
        }
        
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides)
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        self.shape.is_f_contiguous(&self.strides)
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2{
            return Err("Dimension out of range".into());
        }

        let mut strides = self.strides.to_vec();
        strides.swap(dim1, dim2);

        let mut dims = self.shape.dims().to_vec();
        dims.swap(dim1, dim2);
        Ok(Self::new(Shape::from(dims), strides, self.start_offset))
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn transpose() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = shape.stride_contiguous();
        assert_eq!(strides, &[12, 4, 1]);
        let layout = Layout::new(shape, strides, 0);
        let layout = layout.transpose(0, 1).unwrap();
        assert_eq!(layout.shape().dims(), &[3,2,4]);
        assert_eq!(layout.strides(), &[4, 12, 1]);
    }

    #[test]
    fn is_contiguous() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = shape.stride_contiguous();
        let layout = Layout::new(shape, strides, 0);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn is_fortran_contiguous() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = shape.stride_contiguous();
        let layout = Layout::new(shape, strides, 0);
        let layout = layout.transpose(1, 2).unwrap();
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn narrow() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = shape.stride_contiguous();
        let layout = Layout::new(shape, strides, 0);
        let layout = layout.narrow(1, 1, 2).unwrap();
        assert_eq!(layout.shape().dims(), &[2, 2, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert_eq!(layout.start_offset(), 4);
    }

    #[test]
    fn strided_blocks() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = shape.stride_contiguous();
        let layout = Layout::new(shape, strides, 0);
        let blocks = layout.strided_blocks();
        assert_eq!(blocks, StridedBlocks::SingleBlock {
            start_offset: 0,
            len: 24
        });
    }

    #[test]
    fn strided_blocks_single() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = vec![12, 4, 1];
        let layout = Layout::new(shape, strides, 0);
        let blocks = layout.strided_blocks();
        assert_eq!(blocks, StridedBlocks::SingleBlock {
            start_offset: 0,
            len: 24
        });
    }

    #[test]
    fn strided_blocks_multiple() {
        let shape = Shape::from(vec![2, 3, 4]);
        let strides = vec![12, 1, 4];
        let layout = Layout::new(shape, strides, 0);
        let blocks = layout.strided_blocks();
        assert_eq!(blocks, StridedBlocks::MultipleBlocks {
            block_start_index: StridedIndex::new(&[2,3,4], &[12,1,4], 0),
            block_len: 1
        });
    }

    #[test]
    fn strided_blocks_multiple_2() {
        let shape = Shape::from(vec![2, 3, 4]); // strides = [12, 4, 1]
        let strides = vec![1, 4, 12];
        let layout = Layout::new(shape, strides, 0);
        let blocks = layout.strided_blocks();
        assert_eq!(blocks, StridedBlocks::MultipleBlocks {
            block_start_index: StridedIndex::new(&[2,3,4], &[1, 4, 12], 0),
            block_len: 1
        });
    }

}
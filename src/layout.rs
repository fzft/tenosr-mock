use crate::shape::Shape;
use crate::strided_index::StridedBlocks;

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
            todo!("MultipleBlocks")
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

}
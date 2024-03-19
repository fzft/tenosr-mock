#![allow(unused_imports)]
use std::error::Error;
use std::sync::Mutex;

use rand::Rng;
use rayon::prelude::*;

use crate::shape::Shape;
use crate::dtype::Dtype;
use crate::accelerate::dgemm;
use crate::layout::Layout;
use crate::strided_index::StridedBlocks;
use crate::op::{BinaryOpT, UnaryOpT};


#[derive(Debug, Clone)]
pub enum CpuStorage {
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl From<Vec<f64>> for CpuStorage {
    fn from(data: Vec<f64>) -> Self {
        CpuStorage::F64(data)
    }
}

impl CpuStorage {
    pub fn ones(shape: &Shape, dtype: &Dtype) -> Result<CpuStorage, Box<dyn Error>> {
        let elements = shape.elem_count();
        let storage = match dtype {
            Dtype::I64 => {
                let data = vec![1i64; elements];
                CpuStorage::I64(data)
            }
            Dtype::F64 => {
                let data = vec![1f64; elements];
                CpuStorage::F64(data)
            }
            _ => unimplemented!("Not implemented yet")
        };
        Ok(storage)
    }

    pub fn zeros(shape: &Shape, dtype: &Dtype) -> Result<CpuStorage, Box<dyn Error>> {
        let elements = shape.elem_count();
        let storage = match dtype {
            Dtype::I64 => {
                let data = vec![0i64; elements];
                CpuStorage::I64(data)
            }
            Dtype::F64 => {
                let data = vec![0f64; elements];
                CpuStorage::F64(data)
            }
            _ => unimplemented!("Not implemented yet")
        };
        Ok(storage)
    }

    pub fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) {
        match (self, dst) {
            (CpuStorage::F64(src), CpuStorage::F64(dst)) => {
                match src_l.strided_blocks() {
                    StridedBlocks::SingleBlock { start_offset, len } => {
                        let to_copy = (dst.len() - dst_offset).min(len);
                        dst[dst_offset..dst_offset + to_copy].copy_from_slice(&src[start_offset..start_offset + to_copy]);
                    }
                    StridedBlocks::MultipleBlocks { block_start_index, block_len: 1 } => {
                        for (dst_index, src_index) in block_start_index.enumerate(){
                            let dst_index = dst_index + dst_offset;
                            if dst_index >= dst.len() {
                                break;
                            }
                            dst[dst_index] = src[src_index];
                        }
                    }
                    StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
                        unimplemented!("Not implemented yet3")
                    }
                }
            }
            _ => unimplemented!("Not implemented yet")
        }
    }


    pub fn rand_uniform(lo: f64, hi: f64, shape: &Shape, dtype: &Dtype) -> Result<CpuStorage, Box<dyn Error>> {
        let elements = shape.elem_count();
        let mut rng = rand::thread_rng();
        let storage = match dtype {
            Dtype::F64 => {
                let uniform = rand::distributions::Uniform::new(lo, hi);
                let data = (0..elements).map(|_| rng.sample(uniform)).collect();
                CpuStorage::F64(data)
            }
            _ => unimplemented!("Not implemented yet")
        };
        Ok(storage)
    }

    pub fn randn(mean: f64, std: f64, shape: &Shape, dtype: &Dtype) -> Result<CpuStorage, Box<dyn Error>> {
        let elements = shape.elem_count();
        let mut rng = rand::thread_rng();
        let storage = match dtype {
            Dtype::F64 => {
                let normal = rand_distr::Normal::new(mean, std)?;
                let data = (0..elements).map(|_| rng.sample(normal)).collect();
                CpuStorage::F64(data)
            }
            _ => unimplemented!("Not implemented yet")
        };
        Ok(storage)
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            CpuStorage::I64(_) => Dtype::I64,
            CpuStorage::F64(_) => Dtype::F64,
        }
    }

    pub fn from(data: Vec<f64>) -> Result<CpuStorage, Box<dyn Error>> {
        Ok(CpuStorage::F64(data))
    }

    pub fn unary_op<B: UnaryOpT>(&self, layout: &Layout) -> Result<CpuStorage, Box<dyn Error>> {
        match self {
            CpuStorage::F64(data) => {
                let result = unary_map(data, layout, B::f64);
                Ok(CpuStorage::F64(result))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }
    
    pub fn binary_op<B: BinaryOpT>(&self, rhs: &CpuStorage, lhs_l: &Layout, rhs_l: &Layout) -> Result<CpuStorage, Box<dyn Error>> {
        match (self, rhs) {
            (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                let result = binary_map(lhs_l, rhs_l, lhs, rhs, B::f64);
                Ok(CpuStorage::F64(result))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    pub fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<CpuStorage, Box<dyn Error>> {
        match self {
            CpuStorage::F64(data) => {
                let result = data.iter().map(|x| x * mul + add).collect();
                Ok(CpuStorage::F64(result))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }


    #[cfg(not(feature = "accelerate"))]
    pub fn matmul(&self, rhs: &CpuStorage, bmnk: (usize, usize, usize, usize), lhs_layout: &Layout, rhs_layout: &Layout) -> Result<CpuStorage, Box<dyn Error>> {
        match (self, rhs) {
            (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                let (b, m, n, k) = bmnk;
                let result = Mutex::new(vec![0f64; b * m * n]);
                (0..b).into_par_iter().for_each(|bi| {
                    for mi in 0..m {
                        for ni in 0..n {
                            for ki in 0..k {
                                let lhs_index = bi * lhs_layout.strides()[0] + ki * lhs_layout.strides()[1] + mi * k;
                                let rhs_index = bi * rhs_layout.strides()[0] + ni * rhs_layout.strides()[1] + ki * n;
                                let result_index = bi * m * n + mi * n + ni;
                                let mut result_guard = result.lock().unwrap();
                                result_guard[result_index] += lhs[lhs_index] * rhs[rhs_index];
                            }
                        }
                    }
                });
                Ok(CpuStorage::F64(result.into_inner().unwrap()))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    #[cfg(feature = "accelerate")]
    pub fn matmul(&self, rhs: &CpuStorage, bmnk: (usize, usize, usize, usize), lhs_layout: &Layout, rhs_layout: &Layout) -> Result<CpuStorage, Box<dyn Error>> {
        match (self, rhs) {
            (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                let (b, m, n, k) = bmnk;
                let mut result = vec![0f64; b * m * n];
                let lhs_stride = lhs_layout.strides();
                let rhs_stride = rhs_layout.strides();
                let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
                let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
                let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
                let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
                let (lda, transa) = if rhs_m1 == 1 && rhs_m2 == n {
                    (n as i32, b'N')
                } else if rhs_m1 == k && rhs_m2 == 1 {
                    (k as i32, b'T')
                } else {
                    return Err("Invalid layout for rhs".into());
                };

                let (ldb, transb) = if lhs_m1 == 1 && lhs_m2 == k {
                    (k as i32, b'N')
                } else if lhs_m1 == m && lhs_m2 == 1 {
                    (m as i32, b'T')
                } else {
                    return Err("Invalid layout for lhs".into());

                };
                unsafe {
                    dgemm(
                        transa,
                        transb,
                        m as i32,
                        n as i32,
                        k as i32,
                        1.0,
                        lhs,
                        lda as i32,
                        rhs,
                        ldb as i32,
                        0.0,
                        &mut result,
                        n as i32,
                    );
                }
                Ok(CpuStorage::F64(result))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

}

fn unary_map<T: Copy, U: Copy, F: FnMut(T)-> U>(data: &[T], layout: &Layout, mut op: F) -> Vec<U> {
    match layout.strided_blocks() {
        StridedBlocks::SingleBlock {start_offset, len} => {
            data[start_offset..start_offset + len].iter().map(|&x| op(x)).collect()
        }
        _ => unimplemented!("Not implemented yet")
    }
}

fn binary_map<T: Copy, U: Copy, V: Copy, F: FnMut(T, V) -> U>(lhs_l: &Layout, rhs_l: &Layout, lhs: &[T], rhs: &[V], mut op: F) -> Vec<U> {
    match (lhs_l.strided_blocks(), rhs_l.strided_blocks()) {
        (StridedBlocks::SingleBlock {start_offset: lhs_start, len: lhs_len}, StridedBlocks::SingleBlock {start_offset: rhs_start, len: rhs_len}) => {
            let lhs = &lhs[lhs_start..lhs_start + lhs_len];
            let rhs = &rhs[rhs_start..rhs_start + rhs_len];
            lhs.iter().zip(rhs.iter()).map(|(&x, &y)| op(x, y)).collect()
        }
        _ => unimplemented!("Not implemented yet")
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ones() {
        let shape = Shape::from_dims(&[2, 3]);
        let dtype = Dtype::F64;
        let storage = CpuStorage::ones(&shape, &dtype).unwrap();
        match storage {
            CpuStorage::F64(data) => {
                assert_eq!(data, vec![1f64; 6]);
            }
            _ => panic!("Invalid storage type")
        }
    }

    #[test]
    fn test_zeros() {
        let shape = Shape::from_dims(&[2, 3]);
        let dtype = Dtype::F64;
        let storage = CpuStorage::zeros(&shape, &dtype).unwrap();
        match storage {
            CpuStorage::F64(data) => {
                assert_eq!(data, vec![0f64; 6]);
            }
            _ => panic!("Invalid storage type")
        }
    }

    #[test]
    fn test_rand_uniform() {
        let shape = Shape::from_dims(&[2, 3]);
        let dtype = Dtype::F64;
        let storage = CpuStorage::rand_uniform(0.0, 1.0, &shape, &dtype).unwrap();
        match storage {
            CpuStorage::F64(data) => {
                assert_eq!(data.len(), 6);
                for &val in data.iter() {
                    assert!(val >= 0.0 && val < 1.0);
                }
            }
            _ => panic!("Invalid storage type")
        }
    }


    // Check that the random numbers are normally distributed, we should create a large number of random numbers
    #[test]
    fn test_randn() {
        let shape = Shape::from_dims(&[100, 100]);
        let dtype = Dtype::F64;
        let storage = CpuStorage::randn(0.0, 1.0, &shape, &dtype).unwrap();
        match storage {
            CpuStorage::F64(data) => {
                assert_eq!(data.len(), 100 * 100);
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                assert!((mean - 0.0).abs() < 0.1);
                assert!((variance - 1.0).abs() < 0.1);
            }
            _ => panic!("Invalid storage type")
        }
    }
}

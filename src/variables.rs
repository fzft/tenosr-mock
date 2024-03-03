use crate::device::{Device, NdArray};
use crate::dtype::Dtype;
use crate::shape::Shape;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Var(Tensor);


impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::ops::Deref for Var {
    type Target = Tensor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Var {
    
    pub fn new<A: NdArray>(array: A,
                               device: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = Tensor::new_impl(array, device, true)?; // true means it's a variable
        Ok(Self(inner))
    }
    
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: Dtype,
                                 device: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = Tensor::zeros_impl(shape, dtype, device, true)?;
        Ok(Self(inner))
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: Dtype,
                                device: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = Tensor::ones_impl(shape, dtype, device, true)?;
        Ok(Self(inner))
    }
    
    pub fn rand<S: Into<Shape>>(
        mean: f64,
        std: f64,
        s: S,
        device: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = Tensor::randn_impl(mean, std, s, device, true)?;
        Ok(Self(inner))
    }
    
    pub fn randn_uniform<S: Into<Shape>>(
        lo: f64,
        up: f64,
        s: S,
        device: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = Tensor::rand_uniform_impl(lo, up, s, device, true)?;
        Ok(Self(inner))
    }

    pub fn from_tensor(t: &Tensor) -> Self {
        Self(t.clone())
    }
}
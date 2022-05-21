use std::rc::Rc;

use numpy::pyo3::ToBorrowedObject;
use tch::Tensor;



#[derive(Debug, Clone)]
pub struct Detection{
    pub tlwh: [f64;4],
    pub confidence: f64,
    pub feature: Rc<Tensor>,
}


impl Detection{

    pub fn new(tlwh:[f64;4], confidence:f64, feature:Rc<Tensor>) -> Self{
        Detection{
            tlwh,
            confidence,
            feature
        }
    }

    pub fn to_tlbr(&self)-> [f64; 4]{
        let mut ret = self.tlwh.clone();
        ret[2] = ret[2] + ret[0];
        ret[3] = ret[3] + ret[1];
        ret
    }

    pub fn to_xyah(&self) -> [f64; 4]{
        let mut ret = self.tlwh.clone();
        ret[0] += ret[2] / 2.0;
        ret[1] += ret[3] / 2.0 ;
        ret[2] /= ret[3];
        ret
    }
}


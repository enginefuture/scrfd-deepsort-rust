use std::{collections::HashMap, rc::Rc};

use tch::Tensor;


pub fn _nn_cosine_distance(x:Vec<Rc<Tensor>>, y:Vec<Rc<Tensor>>) -> Vec<f64>{

    let mut z = vec![];
    for i in &x{
        let a = i.as_ref() / i.norm();
        let mut sub_z =vec![];
        for j in &y{
            let mut b = j.as_ref() / j.norm();
            let c = a.dot(&b.t_());
            
            sub_z.push(1.0 - f64::from(c.data()))
        }
        z.push(sub_z);
    }

    // println!("{:?}", z);
    
    // let mut norm_a = vec![];
    // for i in x.clone(){
    //     let mut s:f64 = 0.0;
    //     for j in i{
    //         s += j * j;
    //     }
    //     let s = s.sqrt();
    //     norm_a.push(s);
    // }
    
    // let mut norm_b = vec![];
    // for i in y.clone(){
    //     let mut s:f64 = 0.0;
    //     for j in i{
    //         s += j* j;
    //     }
    //     let s = s.sqrt();
    //     norm_b.push(s);
    // }
    

    // let mut a = vec![];
    // for i in 0..x.len(){
    //     let mut sub_a = vec![];
    //     for j in 0..x[i].len(){
    //         let v = x[i][j]/ norm_a[i];
    //         sub_a.push(v );
    //     }
    //     a.push(sub_a);
    // }
    
    // let mut b = vec![];
    // for i in 0..y.len(){
    //     let mut sub_b = vec![];
    //     for j in 0..y[i].len(){
    //         let v = y[i][j] / norm_b[i];
    //         sub_b.push(v);
    //     }
    //     b.push(sub_b);
    // }

    // let mut z = vec![];
    // // 矩阵相乘
    // for i in 0..a.len(){
    //     let mut sub_z = vec![];
    //     for j in 0..b.len(){
    //         let mut sum = 0.;
    //         for k in 0..a[i].len(){
    //             let v = a[i][k] * b[j][k];
    //             sum += v;
    //         }
    //         sub_z.push(1. - sum);
    //     }
    //     // z.push(1. - s);
    //     z.push(sub_z);
    // }
    
    // 求毎列的最小值
    let mut min = vec![];
    for i in 0..y.len(){

        let mut m = z[0][i];
        for j in 0..x.len(){
            if z[j][i] < m{
                m = z[j][i];
            }
        }
        min.push(m);
    }

    min
}

#[derive(Clone)]
pub struct NearestNeighborDistanceMetric{
    pub metric: String,
    pub matching_threshold: f64,
    pub budget: i32,
    pub samples:HashMap<i32,Vec<Rc<Tensor>>>
    
}


impl Default for NearestNeighborDistanceMetric{
    fn default() -> Self {
        Self { matching_threshold: Default::default(), budget: Default::default(), metric: Default::default(), samples: Default::default() }
    }
}


impl NearestNeighborDistanceMetric{

    pub fn new(metric: &str, matching_threshold: f64, budget: i32) -> Self{
        let mut nn = NearestNeighborDistanceMetric::default();
        nn.metric = metric.to_string();
        nn.matching_threshold = matching_threshold;
        nn.budget = budget;
        
        nn.samples = HashMap::new();
        nn
    }

    pub fn partial_fit(&mut self, features: Vec<Rc<Tensor>>, targets: Vec<i32>, active_targets: Vec<i32>){

        for i in 0..targets.len(){
            let feature = features[i].clone();
            let target = targets[i];
            let v = self.samples.get_mut(&target);
            if v.is_none(){
                let mut fs = vec![];
                fs.push(feature);
                self.samples.insert(target, fs);
            }else {
                let v = v.unwrap();
                v.push(feature);
                if self.budget < v.len().try_into().unwrap(){
                    v.remove(0);
                }
            }    
        }

        let mut samples = HashMap::new();

        for k in active_targets{
            samples.insert(k, self.samples[&k].clone());
        }

        self.samples = samples;

    }

    pub fn distance(&mut self, features: Vec<Rc<Tensor>>, targets: Vec<i32>) -> Vec<Vec<f64>>{
        let mut cost_matrix = vec![];
        for i in 0..targets.len(){
            let mut sub_cost = vec![];
            for j in 0..features.len(){
                sub_cost.push(0.);
            }
            cost_matrix.push(sub_cost);
        }

        for i in 0..targets.len(){
            let target = targets[i];
            let sub_cost = self._metric(self.samples[&target].clone(), features.clone());
            
            cost_matrix[i] = sub_cost;
        }

        cost_matrix
    }

    pub fn _metric(&self, x:Vec<Rc<Tensor>>, y:Vec<Rc<Tensor>>) -> Vec<f64>{
        if self.metric.eq("euclidean"){

        }else if self.metric.eq("cosine") {

            return _nn_cosine_distance(x, y);
        }
        vec![0.]
    }

}


#[cfg(test)]
mod tests{

    #[test]
    fn test(){
        // let i:f32 = 4.0;
        // let t = i.sqrt();
        // println!("{:?}", t)
    }
}
use anyhow::Result;
use ndarray as np;
use np::s;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

pub const CHI2INV95: [f64;9] = [
    3.8415,
    5.9915,
    7.8147,
    9.4877,
    11.070,
    12.592,
    14.067,
    15.507,
    16.919
];

#[derive(Clone)]
pub struct KalmanFilter{
    _motion_mat: np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,
    _update_mat: np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,
    _std_weight_position: f64,
    _std_weight_velocity: f64,
}


impl Default for KalmanFilter {
    fn default() -> Self {
        Self { _motion_mat: Default::default(), _update_mat: Default::default(), _std_weight_position: Default::default(), _std_weight_velocity: Default::default()  }
    }
}


impl KalmanFilter {
    pub fn new() ->Self{
        let mut kf = KalmanFilter::default();
        let (ndim, dt) = (4, 1);
        let mut _motion_mat = np::Array2::eye(8);

        for i in 0..ndim{
            let d = _motion_mat.get_mut([i,ndim + i]).unwrap();
            *d = dt as f64;
        }
        kf._motion_mat = _motion_mat;
        let mut _update_mat:np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>> = np::Array2::eye(2*ndim);
        // _update_mat = _motion_mat.slice(s![0..ndim, 0..2*ndim]);
        kf._update_mat = _update_mat.slice(s![0..ndim, 0..2*ndim]).map(|x|*x);

        kf._std_weight_position = 1. / 20.0;
        kf._std_weight_velocity = 1. / 160.0;
        kf
    }

    pub fn initiate(&self, measurement:[f64;4]) -> (np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,){
        
        let mut mean = np::Array::<f64, _>::zeros(8);
        for i in 0..measurement.len(){
            let v = mean.get_mut(i).unwrap();
            *v = measurement[i];
        }

        let std = [
            2.0 * self._std_weight_position * measurement[3],
            2.0 * self._std_weight_position * measurement[3],
            1e-2,
            2.0 * self._std_weight_position * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self._std_weight_velocity * measurement[3],
        ];

        let mut std_mut = np::Array::<f64, _>::zeros((8, 8));
        for i in 0..std.len(){
            let v = std_mut.get_mut((i, i)).unwrap();
            *v = std[i] * std[i];
        }
        (mean, std_mut)
    }

    pub fn predict(&self, mean:&np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, covariance: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,) -> (np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>){

        let std_pos = [
            self._std_weight_position * mean.get(3).unwrap().to_owned(),
            self._std_weight_position * mean.get(3).unwrap().to_owned(),
            1e-2,
            self._std_weight_position * mean.get(3).unwrap().to_owned(),
        ];

        let std_vel = [
            self._std_weight_velocity * mean.get(3).unwrap().to_owned(),
            self._std_weight_velocity * mean.get(3).unwrap().to_owned(),
            1e-5,
            self._std_weight_velocity * mean.get(3).unwrap().to_owned(),
        ];
        
        let mut std = vec![];
        std.append(&mut std_pos.to_vec());
        std.append(&mut std_vel.to_vec());
        
        let mut motion_cov = np::Array::<f64, _>::zeros((8, 8));

        for i in 0..std.len(){
            let v = motion_cov.get_mut((i,i)).unwrap();
            *v = std[i] * std[i];
        }
        let mean = self._motion_mat.dot(mean);
        let covariance = self._motion_mat.dot(covariance);
        let covariance = covariance.dot(&self._motion_mat.t()) + motion_cov;
        (mean, covariance)
    }

    pub fn project(&self, mean: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, covariance: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,) -> (np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>){
        let mean_3 = mean.get(3).unwrap().to_owned();
        let std = [
            self._std_weight_position * mean_3,
            self._std_weight_position * mean_3,
            1e-1,
            self._std_weight_position * mean_3,
        ];

        let mut innovation_cov = np::Array::<f64, _>::zeros((4, 4));
        for i in 0..4{
            let v = innovation_cov.get_mut((i, i)).unwrap();
            *v = std[i] * std[i];
            
        }

        let mean = self._update_mat.dot(mean);

        let covariance = self._update_mat.dot(covariance);
        let covariance = covariance.dot(&self._update_mat.t()) + innovation_cov;

        (mean, covariance)
        
    }
    pub fn update(&self, mean: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, covariance: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>, measurement:[f64;4]) -> Result<(np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,)>{
        let (projected_mean, projected_cov) = self.project(mean, covariance);

        let mut one_dim_projected_cov = vec![];
        for i in 0..4{
            for j in 0..4{
                one_dim_projected_cov.push(projected_cov.get((i, j)).unwrap().to_owned());
            }
        }
        let chol_factor =  cholesky(one_dim_projected_cov, 4);

        // 求x值
        // kalman_gain = scipy.linalg.cho_solve
        let dot_covariance = covariance.dot(&self._update_mat.t());
        let chol_factor_mat = Matrix::new(4, 4, chol_factor);

        let mut kalman_gain = np::Array::<f64, _>::zeros((8, 4));
        for i in 0..8{
            let mut one_dim_covariance = vec![];
            for j in 0..4{

                one_dim_covariance.push(dot_covariance.get((i, j)).unwrap().to_owned());
            }

            let covatiance_mat = Vector::new(one_dim_covariance);

            let x = chol_factor_mat.clone().solve(covatiance_mat)?;
            let y1 = Vector::new(x.data().clone());

            let x1 = chol_factor_mat.clone().solve(y1)?;
            for j in 0..x1.data().len(){
                let v = kalman_gain.get_mut((i, j)).unwrap();
                *v = x1.data()[j];
            }
        }

        // innovation = measurement - projected_mean
        let mut innovation = np::Array::<f64, _>::zeros(4);

        for i in 0..measurement.len(){
            let v = innovation.get_mut(i).unwrap();
            *v = measurement[i] as f64 - projected_mean.get(i).unwrap().to_owned();
        }

        let new_mean= mean + innovation.dot(&kalman_gain.t());

        // new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        let new_covariance = covariance - kalman_gain.dot(&projected_cov).dot(&kalman_gain.t());

        Ok((new_mean, new_covariance))
        
    }

    pub fn gating_distance(
        &self
        , mean: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>
        , covariance: &np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>
        , measurements: Vec<Vec<f64>>
        , only_position:bool
    ) -> Vec<f64>{
        let (mean, covariance) = self.project(&mean, &covariance);

        if only_position{
            // python代码
            // mean, covariance = mean[:2], covariance[:2, :2]
            // measurements = measurements[:, :2]

        }
        let mut one_dim_projected_cov = vec![];
        for i in 0..4{
            for j in 0..4{
                one_dim_projected_cov.push(covariance.get((i, j)).unwrap().to_owned());
            }
        }
        
        let cholesky_factor =  cholesky(one_dim_projected_cov, 4);
        // d = measurements - mean        
        let mut d = vec![];
        for i in 0..measurements.len(){
            let mut sub_d = vec![];
            for j in 0..measurements[i].len(){
                let v = measurements[i][j] - mean.get(j).unwrap().to_owned();
                sub_d.push(v);
            }
            
            d.push(sub_d);
        }
        // 求解方程ax=b
        let mut z = vec![];
        let chol_factor_mat = Matrix::new(4, 4, cholesky_factor);
        for i in 0..d.len(){
            let mut one_dim_covariance = vec![];
            for j in 0..d[0].len(){

                one_dim_covariance.push(d[i][j]);
            }
            // println!("tt1::{:?}", dot_covariance[i]);
            // println!("tt::{:?}", one_dim_covariance);
           

            let covatiance_mat = Vector::new(one_dim_covariance);
            
            let x = chol_factor_mat.clone().solve(covatiance_mat).unwrap();
            // let y1 = Vector::new(x.data().clone());

            // let x1 = chol_factor_mat.clone().solve(y1).unwrap();
            z.push(x.data().clone());

        }

        let mut squared_maha = vec![];
        for l in z{
            let mut sum = 0.;
            for v in l{
                sum += v * v;
            }
            squared_maha.push(sum);
        }

        squared_maha
    }
}

fn cholesky(mat: Vec<f64>, n:usize) -> Vec<f64>{
    let mut res = vec![0.0; mat.len()];
    for i in 0..n{
        for j in 0..(i+1){
            let mut s = 0.0;
            for k in 0..j{
                s += res[i * n + k] * res[j *n + k];
            }
            res[i * n + j] = if i == j {
                (mat[i * n +i] -s).sqrt()
            }else{
                (1.0 / res[j * n + j] * (mat[i * n + j] - s))
            }
        }
    }
    res
}


#[cfg(test)]
mod tests{
    use super::*;

    

    #[test]
    fn test_2(){
        // let mut v = vec![];
        // v.push(1);
        // v.push(2);
        // v.push(3);
        // v.push(4);
        // let t = np::arr1(&v);
        // println!("{:?}", t);
        let mut t = vec![];
        for i in 0..4{
            // let mut t1 = vec![];
            for i in 0..4{
                t.push(i * i);
            }
            // t.push(t1);
        }
        println!("{:?}", t);
        // let t3 = np::arr1(&t).reshape((t.len() / 4,4));
        let t3 = np::rcarr1(&t).reshape((4, 4));
        println!("{:?}", t3);
        
        // println!("{:?}", t3);

    }

    #[test]
    fn test_1(){
        println!("{:?}", 1);
        let mut v = vec![];
        for i in 0..8{
            let mut sv = vec![];
            for j in 0..8{
                sv.push(j as f32);
            }
            v.push(sv);
        }

        // let t = np::arr2(&v[..]);

        // println!("{:?}", t);

    }

    #[test]
    fn test(){

        // let d = np::arr1(&[8, 8]);
        let mut a = np::Array2::eye(8);
        // println!("{:?}", a);
        let t = a.get_mut([0,0]).unwrap();
        *t = 10.0;
        let ad = a.slice(s![0..4, 0..8]).map(|x| *x);
        println!("{:?}", ad);
        
        // let t = [1,2,3,4];
        // let t1 = [5,6,7,8];
        // let mut t2 = vec![];
        // t2.append(&mut t.to_vec());
        // t2.append(&mut t1.to_vec());
        // println!("{:?}", t2);

    }
}
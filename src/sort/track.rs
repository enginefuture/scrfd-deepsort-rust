use std::rc::Rc;

use anyhow::Result;
use tch::Tensor;
use super::{kalman_filter::KalmanFilter, detection::Detection};
use ndarray as np;

const Tentative: i32 = 1;
const Confirmed: i32 = 2;
const Deleted: i32 = 3;

#[derive(Debug, Clone)]
pub struct Track{
    pub mean: np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>,
    pub covariance: np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>,
    pub track_id: i32,
    pub hits: i32,
    pub age: i32,
    pub time_since_update: i32,
    pub state: i32,
    pub features: Vec<Rc<Tensor>>,
    pub _n_init: i32,
    pub _max_age: i32,
}


impl Default for Track {
    fn default() -> Self {
        Self { mean: Default::default(), covariance: Default::default(), track_id: Default::default(), hits: Default::default(), age: Default::default(), time_since_update: Default::default(), state: Default::default(), features: Default::default(), _n_init: Default::default(), _max_age: Default::default() }
    }
}


impl Track{

    pub fn new(mean: np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 1]>>, covariance: np::ArrayBase<np::OwnedRepr<f64>, np::Dim<[usize; 2]>>, track_id: i32, n_init: i32, max_age: i32, feature: Rc<Tensor>) -> Self{
        let mut track = Track::default();
        track.mean = mean;
        track.covariance = covariance;
        track.track_id = track_id;
        track.hits = 1;
        track.age = 1;
        track.time_since_update = 0;
        track.state = Tentative;
        track.features = vec![];

        if feature.size()[0]!= 0{
            track.features.push(feature);
        }
        track._n_init = n_init;
        track._max_age = max_age;
        track
    }

    pub fn to_tlwh(&self) -> Vec<f64>{
        let mut ret = vec![];
        for i in 0..4{
            ret.push(self.mean.get(i).unwrap().to_owned());
        }
        ret[2] = ret[2] * ret[3];
        ret[0] = ret[0] - ret[2] / 2.0;
        ret[1] = ret[1] - ret[3] / 2.0;
        ret
    }
    pub fn predict(&mut self, kf: &mut KalmanFilter){

        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);

        self.mean = mean;
        self.covariance = covariance;
        self.age += 1;
        self.time_since_update += 1;

    }

    pub fn update(&mut self, kf: &mut KalmanFilter, detection:Detection) -> Result<()>{
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, detection.to_xyah())?;

        self.mean = mean;
        self.covariance = covariance;
        self.features.push(detection.feature);
        self.hits += 1;

        self.time_since_update = 0;
        if self.state == Tentative && self.hits >= self._n_init{

            self.state = Confirmed;
        }

        Ok(())
    }
    pub fn mark_missed(&mut self){
        if self.state == Tentative{
            self.state = Deleted;
        }else if self.time_since_update > self._max_age {
            self.state = Deleted;
        }
    }

    pub fn is_tentative(&self) -> bool{
        self.state == Tentative
    }

    pub fn is_confirmed(&self) ->bool{
        self.state == Confirmed
    }

    pub fn is_deleted(&self) -> bool{
        self.state == Deleted
    }

}


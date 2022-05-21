use super::{nn_matching::NearestNeighborDistanceMetric, kalman_filter::KalmanFilter, detection::Detection, track::Track, linear_assignment::{matching_cascade, min_cost_matching}};

#[derive(Clone)]
pub struct Tracker{
    pub metric: NearestNeighborDistanceMetric,
    pub max_iou_distance: f64,
    pub max_age: i32,
    pub n_init: i32,
    pub kf: KalmanFilter,
    pub tracks: Vec<Track>,
    pub _next_id: i32,
}

impl Default for Tracker {
    fn default() -> Self {
        Self { metric: Default::default(), max_iou_distance: Default::default(), max_age: Default::default(), n_init: Default::default(), kf: Default::default(), tracks: Default::default(), _next_id: Default::default() }
    }
}


impl Tracker {
    
    pub fn new(
        metric:NearestNeighborDistanceMetric
        , max_iou_distance:f64
        , max_age: i32
        , n_init: i32
    ) -> Self{
        let mut tracker = Tracker::default();
        tracker.metric = metric;
        tracker.max_iou_distance = max_iou_distance;
        tracker.max_age = max_age;
        tracker.n_init = n_init;
        tracker.kf = KalmanFilter::new();
        tracker.tracks = Vec::default();
        tracker._next_id = 1;
        tracker
    }

    pub fn predict(&mut self){

        for mut track in &mut self.tracks{
            track.predict(&mut self.kf);
        }
    }

    pub fn update(&mut self, detections: Vec<Detection>){

        let (mut matches, mut unmatched_tracks, mut unmatched_detections) = 
                                self._match(detections.clone())
        ;
        // matches.dedup();
        // unmatched_tracks.dedup();
        // unmatched_detections.dedup();
        println!("matches::{:?}", matches);
        println!("unmatched_tracks::{:?}", unmatched_tracks);
        println!("unmatched_detections::{:?}", unmatched_detections);

        // 更新track 
        for (track_idx, detection_idx) in matches{      
      
            self.tracks[track_idx].update(
                &mut self.kf, detections[detection_idx].clone());
        }
        for track_idx in unmatched_tracks{
            self.tracks[track_idx].mark_missed();
        }

        for detection_idx in unmatched_detections{
            self._initiate_track(detections[detection_idx].clone());
        }
        let mut tracks = vec![];
        for t in self.tracks.clone(){
            if !t.is_deleted(){
                tracks.push(t);
            }
        }
        self.tracks = tracks;

        let mut active_targets = vec![];
        for t in self.tracks.clone(){
            if t.is_confirmed(){
                active_targets.push(t.track_id);
            }
        }

        let mut features = vec![];
        let mut targets = vec![];
        for track in &mut self.tracks{
            if !track.is_confirmed(){
                continue;
            }
            for t in track.features.clone(){
                features.push(t.clone());
                targets.push(track.track_id);
            }
            track.features.clear();
        }
        self.metric.partial_fit(features, targets, active_targets);
    }

    pub fn _match(&mut self, detections: Vec<Detection>) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>){
        
        let mut confirmed_tracks = vec![];
        let mut unconfirmed_tracks = vec![];
        for i in 0..self.tracks.len(){
            let t = &self.tracks[i];
            if t.is_confirmed(){
                confirmed_tracks.push(i);
            }
            
            if !t.is_confirmed(){
                unconfirmed_tracks.push(i)
            }
        }

        let (matches_a, mut unmatched_tracks_a, unmatched_detections) = 
            matching_cascade(self, self.metric.matching_threshold, self.max_age,
                 self.tracks.clone(), detections.clone(), confirmed_tracks, vec![])
        ;
        let mut iou_track_candidates = unconfirmed_tracks;
        
        for k in unmatched_tracks_a.clone(){
            if self.tracks[k].time_since_update == 1{
                iou_track_candidates.push(k);
            }
        }

        let mut ind_v = vec![];
        for k in 0..unmatched_tracks_a.len(){
            
            if self.tracks[unmatched_tracks_a[k]].time_since_update == 1{
                ind_v.push(k);
            }
        }
        ind_v.reverse();
        for k in ind_v{
            unmatched_tracks_a.remove(k);
        }

        let (matches_b, unmatched_tracks_b,mut  unmatched_detections) = 
            min_cost_matching(self, "iou_cost".to_string(),self.max_iou_distance, self.tracks.clone(), detections.clone(), iou_track_candidates, unmatched_detections)
        ;

        let mut matches = vec![];
        for i in matches_a{
            matches.push(i);
        }
        for i in matches_b{
            matches.push(i);
        }

        let mut unmatched_tracks = vec![];
        for i in unmatched_tracks_a{
            unmatched_tracks.push(i);
        }
        for i in unmatched_tracks_b{
            unmatched_tracks.push(i);
        }
        // 去重
        unmatched_tracks.dedup();
        matches.dedup_by(|a, b| a.1==b.1);

        (matches, unmatched_tracks, unmatched_detections)
    }

    pub fn _initiate_track(&mut self, detection: Detection){
       

        let (mean, covariance) = self.kf.initiate(detection.to_xyah());

        let track = Track::new(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature);
        self.tracks.push(track);
        self._next_id += 1;
    }
}
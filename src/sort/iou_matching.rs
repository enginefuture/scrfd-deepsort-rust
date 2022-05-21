use super::{
    detection::Detection
    , track::Track
    , linear_assignment::INFTY_COST
};


fn iou(bbox: Vec<f64>, candidates: Vec<[f64; 4]>) -> Vec<f64>{
    
    let mut bbox_tl = vec![];
    let mut bbox_br = vec![];
    for i in 0..2{
        bbox_tl.push(bbox[i]);
        bbox_br.push(bbox[i] + bbox[i+2]);
    }
    

    let mut candidates_tl = vec![];
    let mut candidates_br = vec![];

    for i in 0..candidates.len(){
        let mut sub_tl = vec![];
        let mut sub_br = vec![];
        for j in 0..2{

            sub_tl.push(candidates[i][j]);
            sub_br.push(candidates[i][j] + candidates[i][j + 2]);

        }
        candidates_tl.push(sub_tl);
        candidates_br.push(sub_br)
    }
    let mut tl = vec![];
    for i in 0..candidates_tl.len(){
        let mut sub_tl = vec![];
        if bbox_tl[0] > candidates_tl[i][0]{
            sub_tl.push(bbox_tl[0]);
        }else {
            sub_tl.push(candidates_tl[i][0]);
        }
        if bbox_tl[1] > candidates_tl[i][1]{
            sub_tl.push(bbox_tl[1]);
        }else {
            sub_tl.push(candidates_tl[i][1]);
        }
        tl.push(sub_tl);
    }
    let mut br = vec![];
    for i in 0..candidates_br.len(){
        let mut sub_br = vec![];
        if bbox_br[0] < candidates_br[i][0]{
            sub_br.push(bbox_br[0]);
        }else {
            sub_br.push(candidates_br[i][0]);
        }

        if bbox_br[1] < candidates_br[i][1]{
            sub_br.push(bbox_br[1]);
        }else {
            sub_br.push(candidates_br[i][1]);
        }
        br.push(sub_br);
    }

    let mut wh = vec![];
    for i in 0..br.len(){
        let mut sub_wh = vec![];
        for j in 0..br[i].len(){
            if 0. > br[i][j] - tl[i][j]{
                sub_wh.push(0.);
            }else{
                sub_wh.push(br[i][j] - tl[i][j]);
            }
        }
        wh.push(sub_wh);
    }
   
    
    let mut area_intersection = vec![];
    for i in 0..wh.len(){
        let mut sum = 1.0;
        for v in wh[i].clone(){
            sum *= v;
        } 
        area_intersection.push(sum);
    }

    let mut area_bbox = 1.0;
    for i in 0..2{
        area_bbox *= bbox[i+2];
    }

    let mut area_candidates = vec![];
    for i in 0..candidates.len(){
        let mut sum = 1.0;
        for j in 0..candidates[i].len() - 2{

            sum *= candidates[i][j+2];
        } 
        area_candidates.push(sum);
    }


    let mut area = vec![];
    for i in 0..area_intersection.len(){
        let v = area_intersection[i] / (area_bbox + area_candidates[i] - area_intersection[i]);
        area.push(v);
    }

    area
}


pub fn iou_cost(
    tracks: Vec<Track>
    , dets: Vec<Detection>
    , mut track_indices: Vec<usize>
    , mut detection_indices: Vec<usize>
) -> Vec<Vec<f64>>{
    // None
    if track_indices.len() < 0{
        for i in 0..tracks.len(){
            track_indices.push(i);
        }
    }
    
    if detection_indices.len() < 0{
        for i in 0..dets.len(){
            detection_indices.push(i);
        }
    }

    let mut cost_matrix = vec![];
    for _ in 0..track_indices.len(){
        let mut sub_cost = vec![];
        for _ in 0..detection_indices.len(){
            sub_cost.push(0.);
        }
        cost_matrix.push(sub_cost);
    }

    for row in 0..track_indices.len(){
        let track_idx = track_indices[row];
        if tracks[track_idx].time_since_update > 1{
            for i in 0..cost_matrix[row].len(){
                cost_matrix[row][i] = INFTY_COST;
            }
            continue;
        }

        let bbox = tracks[track_idx].to_tlwh();

        let mut candidates = vec![];
        for i in detection_indices.clone(){
            let v = dets[i].tlwh;
            candidates.push(v);
        }

        let area = iou(bbox.clone(), candidates.clone());

        for i in 0..cost_matrix[row].len(){
            cost_matrix[row][i] = 1. - area[i];
        }
    }
    cost_matrix
}

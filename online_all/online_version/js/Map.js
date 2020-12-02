//organize the code 


//color constants
const grey = 0xFAF7F6;
const black = 0x000000;
const green = 0xA2EF4C;
//window setting for now
const width = 800 //1000;
const height = 600 //900

//city/distance list for debug
const locations = [[200,150],[270,90],[460,200],[300,200],[450,230],[378,229]];
const dis_matrix = [[0.00,92.19544457,264.7640459,111.80339887,262.48809497,194.74342094],
                    [92.19544457,0.00,219.544984,114.01754251,228.03508502,176.02556632],
                    [264.7640459,219.544984,0.00,160.00000000,31.6227766,86.97700846],
                    [111.80339887,114.01754251,160.00000000,0.00,152.97058541,83.21658489],
                    [262.48809497,228.03508502,31.6227766,152.97058541,0.00,72.00694411],
                    [194.74342094,176.02556632,86.97700846,83.21658489,72.00694411,0.00]];

//map class (map load, data saving, user check)
export default class Map{
  constructor(scene, map_content,trl_id,blk,map_id){
    this.load_map(map_content,map_id);
    this.exp_data_init(scene,blk,trl_id,map_id);
  }

  //load json map into javascript
  //uncommented one are for local debug
  load_map(map_content, map_id){
    // this.loadmap = map_content[map_id];
    // this.order = NaN;
    //
    // this.N = this.loadmap['N'];
    // this.radius = 5; //radius of city
    //
    // this.total = this.loadmap['total'];
    // this.budget_remain = this.loadmap['total'];
    //
    // this.R = this.loadmap['R'];
    // this.r = this.loadmap['r'];
    // this.phi = this.loadmap['phi'];
    // this.x = this.load_map['x'].map(x => x[0] + width/2);
    // this.y = this.load_map['y'].map(x => x[1] + height/2);
    // this.xy = this.x.map(x => [x,this.y[this.x.indexOf(x)]]);
    this.xy = map_content; // for debug purpose
    this.city_start = map_content[0]; //for debug
    // this.distance = this.loadmap['distance'];
    this.distance = dis_matrix; //for debug purpose
    // this.city_start = [200,150]; //for debug same as the first one in the list
    // console.log(this.xy); //join list is not working

    //generate circle map parameters
    this.N = 6; //totall city number, including start
    this.radius = 10; //radius of city
    this.total = 400; //total budget
    this.budget_remain = 400; //remaining budget

    this.R = 400*400; //circle radius' sqaure
  }

  //------------DATA--COLLECTION--------------------------------------------------
  exp_data_init(scene, blk, trl_id, map_id){
    this.blk = [blk];
    this.trl = [trl_id];
    this.mapid = [map_id];
    this.cond = [2]; //condition, road basic

    //let time = scene.Input.Pointer.downTime; bug
    //this.time = [Math.round(time/1000,2)]; bug
    //double check mouse click time

    //this.pointer = new scene.Input.Pointer();
    this.mouse_x = scene.input.mousePointer.x;
    this.mouse_y = scene.input.mousePointer.y;
    //double check mouse saving here
    this.pos = [[this.mouse_x,this.mouse_y]];
    this.click = [0];  //click indicator
    this.undo_press = [0];

    this.choice_dyn = [0];
    this.choice_locdyn = [this.city_start];
    this.choice_his = [0];
    this.choice_loc = [this.city_start];

    this.budget_dyn = [this.total];
    this.budget_his = [this.total];

    this.n_city = [0]; //number of cities connected
    this.check = 0; //indicator showing if people make valid choice

    this.check_end_ind = 0;
  }

  exp_data(mouse, time, blk, trl_id, map_id){
    this.blk.push(blk);
    this.trl.push(trl_id);
    this.mapid.push(map_id);
    this.cond.push(2);
//    this.time.push(time); bug
    this.pos.push(mouse);
    this.click.push(1);
    this.undo_press.push(0);

    this.choice_dyn.push(this.index);
    this.choice_locdyn.push(this.city);
    this.choice_his.push(this.index);
    this.choice_loc.push(this.city);

    this.budget_dyn.push(this.budget_remain);
    this.budget_his.push(this.budget_remain);

    this.n_city.push(this.n_city[this.n_city.length-1]+1);
    this.check = 0; //change choice indicator after saving them

    delete this.index;
    delete this.city;
  }

  static_data(mouse, time, blk, trl_id, map_id){
    this.blk.push(blk);
    this.trl.push(trl_id);
    this.mapid.push(map_id);
    this.cond.push(2);
    //this.time.push(time); bug 
    this.pos.push(mouse);
    this.click.push(0);
    this.undo_press.push(0);

    this.choice_his.push(this.choice_dyn[this.choice_dyn.length-1]);
    this.choice_loc.push(this.choice_locdyn[this.choice_locdyn.length-1]);
    this.budget_his.push(this.budget_dyn[this.budget_dyn.length-1]);

    this.n_city.push(this.n_city[this.n_city.length-1]);
  }

  //---------Check---User---Input-------------------------------------------------
  make_choice(mouse_x,mouse_y){
  //do not evaluate the starting point
    for (var i=1; i<this.xy.length; i++){
      this.mouse_distance = Math.hypot(this.xy[i][0]-mouse_x,
      this.xy[i][1]-mouse_y);
      //currently entering input.pointer, not useing the created pointer object
      //this.xy = locations

      // 5 is based on visual testing, not based on radius
      // check whether is close to city locations && this city hasn't been chosen yet
      if (this.mouse_distance <= 5 && this.choice_dyn.includes(i)==false){
        this.index = i; //index of chosen city
        this.city = this.xy[i];//location of chosen city
        this.check = 1; //indicator showing people made a valid choice
      };
    };
  }

 budget_update(){
  //get distance from current choice to previous choice
  let dist = this.distance[this.index][this.choice_dyn[this.choice_dyn.length-1]];
  this.budget_remain = this.budget_dyn[this.budget_dyn.length-1] - dist;
  }

  //check if trial end, not all dots are evaluated right
 check_end(){
    let distance_copy = this.distance[this.choice_dyn[this.choice_dyn.length-1]].slice();
    // copy distance list for current city
    // take note of the const i of sytax
    for (const i of this.choice_dyn){
        distance_copy[i] = 0;
    };

    if (distance_copy.some(i =>
      i < this.budget_dyn[this.budget_dyn.length-1] && i != 0)){
      return true;
    }else{
      return false;
      };
  }

}
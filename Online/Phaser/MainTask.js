//this create context/scene seperatly to be called in the main file

//color constants
const grey = 0xFAF7F6;
const black = 0x000000;
const green = 0xA2EF4C;
const width = 800 //1000;
const height = 600 //900

//city/response list for debug
//no load function yet
const locations = [[200,150],[270,90],[460,200],[300,200],[450,230],[378,229]];
//6x6 distance matrix manuall intered from python code
const dis_matrix = [[0.00,92.19544457,264.7640459,111.80339887,262.48809497,194.74342094],
                    [92.19544457,0.00,219.544984,114.01754251,228.03508502,176.02556632],
                    [264.7640459,219.544984,0.00,160.00000000,31.6227766,86.97700846],
                    [111.80339887,114.01754251,160.00000000,0.00,152.97058541,83.21658489],
                    [262.48809497,228.03508502,31.6227766,152.97058541,0.00,72.00694411],
                    [194.74342094,176.02556632,86.97700846,83.21658489,72.00694411,0.00]];

class MainTask extends Phaser.Scene {
    constructor() {
        super('MainTask');
    }
    preload()
    {

    }
    create(){
        console.log("Ready!");
        //call your major functions
        this.load_map();
        this.exp_data_init(1,1,1);
        this.budget();
        this.cities();
        this.scorebar();

    };

    //load json map into javascript
    load_map(map_content, map_id){
        //this.loadmap = map_content[map_id];
        this.order = NaN;

        //this.N = this.loadmap['N'];
        // this.radius = 5;

        //this.total = this.loadmap['total'];
        //this.budget_remain = this.loadmap['total'];

        //this.R = this.loadmap['R'];
        //this.r = this.loadmap['r'];
        //this.phi = this.loadmap['phi'];
        // this.x = this.load_map['x'].map(x => x[0] + width/2);
        // this.y = this.load_map['y'].map(x => x[1] + height/2);
        // this.xy = this.x.map(x => [x,this.y[this.x.indexOf(x)]]);
        this.city_start = [200,150]; //for debug same as the first one in the list
        // console.log(this.xy); //not working join list is working

        //generate circle map parameters
        this.N = 20; //totall city number, including start
        this.radius = 10; //radius of city
        this.total = 400; //total budget
        this.budget_remain = 400; //remaining budget

        this.R = 400*400; //circle radius' sqaure
    };

//------------DATA--COLLECTION--------------------------------------------------
    exp_data_init(blk, trl_id, map_id){
      this.blk = [blk];
      this.trl = [trl_id];
      this.mapid = [map_id];
      this.cond = [2]; //condition

      let time = Phaser.Input.Pointer.downTime;
      this.time = [Math.round(time/1000,2)];
      // console.log(this.time);
      //mouse click time, double check

      this.pointer = this.input.activePointer;
      this.pos = [[this.pointer.x,this.pointer.y]];
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
    };

    exp_data(mouse, time, blk, trl_id, map_id){
      this.blk.push(blk);
      this.trl.push(trl_id);
      this.mapid.push(map_id);
      this.cond.push(2);
      this.time.push(time);
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

      //need to double check this delete function
      delete this.index;
      delete this.city;
    };

    static_data(mouse, time, blk, trl_id, map_id){
      this.blk.push(blk);
      this.trl.push(trl_id);
      this.mapid.push(map_id);
      this.cond.push(2);
      this.time.push(time);
      this.pos.push(mouse);
      this.click.push(0);
      this.undo_press.push(0);

      this.choice_his.push(this.choice_dyn[this.choice_dyn.length-1]);
      this.choice_loc.push(this.choice_locdyn[this.choice_locdyn.length-1]);
      this.budget_his.push(this.budget_dyn[this.budget_dyn.length-1]);

      this.n_city.push(this.n_city[this.n_city.length-1]);
    };
//-----------------------------------------------------------------------------

    //so far these are simple function, no agruaments yet
    cities(){
        //create visuals and define style
        this.circle = this.add.graphics();
        this.circle.fillStyle(grey,.5);
        // this.circle.fillCircle(30,50,6);

        //drawing all cities from the map list
        for (var i=1; i<locations.length; i++){
          this.x = locations[i][0];
          this.y = locations[i][1];
          let city = this.circle.fillCircle(this.x,this.y,6);
        };

        //drawing the starting city
        let start = this.circle.fillCircle(this.city_start[0],this.city_start[1],6);
    };

    road(){
      //function name and this.name don't use the same
      //otherwise lead to naming bug

      //create road and define style
      this.line = this.add.graphics();
      this.line.lineStyle(4, grey, 1.0);

      // draw the connected cities from "mmap.choice_locdyn"
      // double check i<= this.choice_locdyn.length
      for (var i=0; i<this.choice_locdyn.length-1; i++){
          let line = new Phaser.Geom.Line(
          this.choice_locdyn[i][0],this.choice_locdyn[i][1],
          this.choice_locdyn[i+1][0],this.choice_locdyn[i+1][1]);
          this.line.strokeLineShape(line);
      };
    };

    budget(){
      //create budget line and define style
      this.budget_line = this.add.graphics();
      this.budget_line.lineStyle(4, green, 1.0);
      //mouse input setup
      //or this.pointer.x
      this.mouse_x = game.input.mousePointer.x;
      this.mouse_y = game.input.mousePointer.y;

      //current city loc: mmap.choice_locdyn[-1][0]
      //JS negative index is different
      let x = this.choice_locdyn[this.choice_locdyn.length - 1][0];
      let y = this.choice_locdyn[this.choice_locdyn.length - 1][1];
      //budget follow mouse
      let cx = this.pointer.x - x;
      let cy = this.pointer.y - y;
      let radians = Math.atan2(cy,cx);
      //mmap.budget_dyn[-1]
      this.budget_pos_x = x + this.budget_dyn[this.budget_dyn.length - 1] * Math.cos(radians);
      // console.log(this.budget_dyn);
      this.budget_pos_y = y + this.budget_dyn[this.budget_dyn.length - 1] * Math.sin(radians);
      //console.log(this.budget_pos_x); Nan here debug
      //draw budget line
      let line = new Phaser.Geom.Line(x,y,this.budget_pos_x,this.budget_pos_y);
      this.budget_line.strokeLineShape(line);
    };

//-------Score--Bar----------------------------------------------------------------------
    scorebar(){
      //score bar parameters
      this.width = 100 //1000;
      this.height = 400 //480;
      this.box = 12;
      this.top = 50 //200; //distance to screen top

      this.box_center(); //center for labels
      this.incentive(); //calculate incentive: N^2
      // this.indicator(mmap)
      this.indicator(); //incentive score indicator, merged with older arrow function
      this.number();
    };

    box_center(){
      this.box_height = this.height / this.box
      this.center_list = []
      this.uni_height = this.box_height / 2
      this.x = this.width / 2 + 600  //larger the number, further to right, 1300

      for (var i=0; i<this.box; i++){
        //double check here, maybe bug
        // const box_y = i * this.box_height + this.uni_height;
        let loc = [this.x, i * this.box_height + this.uni_height];
        this.center_list.push(loc);
      };
    };

    incentive(){
      this.score = Array.from(Array(this.box).keys());
      this.incentive_score = [];
      for (let i of this.score){
        i = (i**2) * 0.01;
        this.incentive_score.push(i);
      };
    };

    indicator(){
      //mmap as argument
      this.indicator_loc = this.center_list[this.n_city[this.n_city.length-1]];
      //mmap.n_city[-1]
      this.indicator_loc_best = this.center_list[Math.max(this.n_city)];
      //mmap.n_city;

      //create triagle arrow and define style
      this.triangle = this.add.graphics();
      this.triangle.fillStyle(grey);

      //arrow parameter
      let point = [this.indicator_loc[0] - 30, this.indicator_loc[1]+this.top+10];
      let v2 = [point[0] - 10, point[1] + 10];
      let v3 = [point[0] - 10, point[1] - 10];
      this.triangle.fillTriangle(point[0], point[1], v2[0],v2[1],v3[0],v3[1]);
    };

    //drawing score Bar
    number(){
      //create rectangle and define style
      this.rect = this.add.graphics();

      let left = this.center_list[0][0] - 25;
      let hex_c = [0x66CC66,0x74C366,0x82B966,0x90B066,
                    0x9EA766,0xAC9E66,0xB99466,0xC78B66,
                    0xD58266,0xE37966,0xF16F66,0xFF6666] //color list

      for (var i=0; i<this.box; i++){
        let loc = this.center_list[i];
        let text = this.incentive_score[i];

        //score bar outline
        this.rect.fillStyle(grey);
        this.rect.fillRect(left,loc[1]+this.top-this.uni_height,
        this.width,this.box_height);

        //score bar fill
        this.rect.fillStyle(hex_c[i]);
        this.rect.fillRect(left,loc[1]+this.top-this.uni_height+2,
        this.width,this.box_height);

        this.add.text(loc[0],loc[1]+this.top,text);
      };

      // scorebar title
      this.add.text(this.center_list[0][0]-20,this.center_list[0][1]+this.top-50,
     'Bonus in dollars');
    };

//-----------------------------------------------------------------------------


//---------Check---User---Input-------------------------------------------------
    make_choice(pointer){
      //do not evaluate the starting point
      for (var i=1; i<locations.length; i++){
        this.mouse_distance = Math.hypot(locations[i][0]-this.pointer.x,
          locations[i][1]-this.pointer.y);

        // 5 is based on visual testing, not based on radius
        // check whether is close to city locations &&
        // this city hasn't been chosen yet
        if (this.mouse_distance <= 5 && this.choice_dyn.includes(i)==false){
          this.index = i; //index of chosen city
          this.city = locations[i];
          // this.city = this.xy[i]; //location of chosen city
          this.check = 1; //indicator showing people made a valid choice
          // console.log("4.enough budget to connect!")
        };
      };
    };

    budget_update(){
      //get distance from current choice to previous choice
      let dist = dis_matrix[this.index][this.choice_dyn[this.choice_dyn.length-1]];
      this.budget_remain = this.budget_dyn[this.budget_dyn.length-1] - dist;
    };

    //check if trial end, not all dots are evaluated right
    check_end(){
      let distance_copy = dis_matrix[this.choice_dyn[this.choice_dyn.length-1]].slice();
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
      };

//--------GAME--LOOP------------------------------------------------------------
    update(){
      this.add.text(20,20,"Road Construction");
      //clear as a function to update per frame
      this.budget_line.clear();
      this.triangle.clear();
      this.budget();
      this.scorebar();
      // this.number();


      this.input.on('pointerdown', function (pointer){
        //maybe change to button up?
        //this.click[-1] = 1
        if (this.pointer.leftButtonDown()){
            if (this.check_end()){
              this.make_choice();
              if (this.check == 1){
                  this.budget_update();
                  this.exp_data(this.pointer,1,1,1,1);
                  this.road();
              }else{
                this.static_data(this.pointer,1,1,1,1);
              };
            }else{
              this.add.text(20,50,"Trial End");
            };
        };
      }, this);
    };
}

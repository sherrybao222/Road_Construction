//this create context/scene seperatly to be called in the main file

//color constants
const grey = 0xFAF7F6;
const green = 0xA2EF4C;
const width = 1000;
const  height = 900

//city/response list for debug
//no load function yet
const locations = [[200,150],[270,90],[460,200],[300,200],[450,230],[378,229]];

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
        this.city_start = [100,100]; //for debug
        // console.log(this.xy); //not working join list is working

        //generate circle map parameters
        this.N = 20; //totall city number, including start
        this.radius = 10; //radius of city
        this.total = 400; //total budget
        this.budget_remain = 400; //remaining budget

        this.R = 400*400; //circle radius' sqaure
    };

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

    make_choice(pointer){
      for (var i=1; i<locations.length; i++){
        this.mouse_distance = Math.hypot(locations[i][0]-this.pointer.x,
          locations[i][1]-this.pointer.y);

        // 5 is based on visual testing, not based on radius
        if (this.mouse_distance <= 5 && this.choice_dyn.includes(this.choice_dyn[i])==false){
          // choice_dyn.include bug, already included but haven't click
          this.index = i; //index of chosen city
          //no distance matrix
          //online calculation of distance using map list and pointer
          this.choice_distance = Phaser.Math.Distance.Between(
                                // locations[this.choice_dyn.length-1][0],
                                // locations[this.choice_dyn.length-1][1],
                                // previous city
                                this.choice_locdyn[this.choice_locdyn.length-1][0],
                                this.choice_locdyn[this.choice_locdyn.length-1][1],
                                // this.city[0],this.city[1]);
                                //chosen city
                                locations[this.index][0], locations[this.index][1]);

          // console.log(this.budget_remain);
          // console.log("budget list: "+ this.budget_dyn);
          // console.log("current distance: " + this.choice_distance);
          console.log("correct city & haven't choosen");
          console.log("distance is " + this.choice_distance);

          if (this.choice_distance <= this.budget_remain){
            this.city = locations[i];
            // this.city = this.xy[i]; //location of chosen city
            this.check = 1; //indicator showing people made a valid choice
            console.log("enough budget to connect!")
          };
        };
      };
    };

    budget_update(){
      let remain = this.budget_remain - this.choice_distance;
      // this.budget_remain[this.budget_remain.length-1] - this.choice_distance;
      //             // budget_dyn[budget_dyn.length-1];
      this.budget_remain = remain;
      // console.log(this.choice_distance);
      // console.log(budget_remain);
    };

    update(){
      this.add.text(20,20,"Road Construction");
      //destroy as a function to update per frame
      this.budget_line.destroy();
      this.budget();

      this.input.on('pointerdown', function (pointer){
        if (this.pointer.leftButtonDown()){
            console.log(this.check);
            this.make_choice();
            if (this.check == 1){
                this.budget_update();
                this.exp_data(this.pointer,1,1,1,1);
                console.log(this.choice_his);
                //seems like data saved multiple times for one click
                this.road();
            };
        };
      }, this);
    };
}

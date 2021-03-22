/*
This is 2020 July most up to date working version of road construction task
including the undo function. For a more detailed note, commend, and reminder,
see the MainTask file. This is the file that will be run when open the html
file in Chrome. 
*/

//import the Map class module (mostly copy pasted, but made some modification for phaser features)
//need to double check scene parameters in the Map class 
import Map from "/scripts/Map.js";

//color constants
const grey = 0xFAF7F6;
const black = 0x000000;
const green = 0xA2EF4C;
//window setting
const width = 800 //1000;
const height = 600 //900

const locations = [[200,150],[270,90],[460,200],[300,200],[450,230],[378,229]];
const dis_matrix = [[0.00,92.19544457,264.7640459,111.80339887,262.48809497,194.74342094],
                    [92.19544457,0.00,219.544984,114.01754251,228.03508502,176.02556632],
                    [264.7640459,219.544984,0.00,160.00000000,31.6227766,86.97700846],
                    [111.80339887,114.01754251,160.00000000,0.00,152.97058541,83.21658489],
                    [262.48809497,228.03508502,31.6227766,152.97058541,0.00,72.00694411],
                    [194.74342094,176.02556632,86.97700846,83.21658489,72.00694411,0.00]];

//creat a single trial under Map class
//let trial = new Map(locations,1,1,1);

//Phaser scene template: filename = constructor super(filename), add the
//same name to the scene lists in script.js, and add script path in html

export default class RCundo extends Phaser.Scene {
    constructor() {
        super('RCundo');
    }
    preload()
    {

    }
    create(){
        console.log("Road Undo Ready!");
        this.time;

        //create undo key press on Z
        this.keyZ = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.Z);
        this.trial = new Map(this,locations,1,1,1); //new trial object from Map Class
        this.draw_map(this.trial,this.input.mousePointer.x,this.input.mousePointer.y);
    }

    //-------Map--Visualization--Functions--------------------------------------
    //this call all the basic map setup
    draw_map(mmap,mouse_x,mouse_y){
      this.budget(mmap,mouse_x,mouse_y);
      this.cities(mmap);
      this.scorebar(mmap);
      this.road(mmap);
    }

    //individual component of map setup
    cities(mmap){
      //create city and define style
      this.circle = this.add.graphics();
      this.circle.fillStyle(grey,.5);

      for (var i=1; i<mmap.xy.length; i++){
        this.x = mmap.xy[i][0];
        this.y = mmap.xy[i][1];
        let city = this.circle.fillCircle(this.x,this.y,6);
      };

      //drawing the starting city
      let start = this.circle.fillCircle(mmap.city_start[0],mmap.city_start[1],6);
    }

    road(mmap){
      //function name and this.name don't use the same >> otherwise lead to naming bug
      //create road and define style
      this.line = this.add.graphics();
      this.line.lineStyle(4, grey, 1.0);

      for (var i=0; i<mmap.choice_locdyn.length-1; i++){
          let line = new Phaser.Geom.Line(
          mmap.choice_locdyn[i][0],mmap.choice_locdyn[i][1],
          mmap.choice_locdyn[i+1][0],mmap.choice_locdyn[i+1][1]);
          this.line.strokeLineShape(line);
      };
    }

    budget(mmap,mouse_x,mouse_y){
      //create budget line and define style
      this.budget_line = this.add.graphics();
      this.budget_line.lineStyle(4, green, 1.0);
      // mouse input setup or this.pointer.x
      // this.mouse_x = game.input.mousePointer.x;
      // this.mouse_y = game.input.mousePointer.y;

      let x = mmap.choice_locdyn[mmap.choice_locdyn.length - 1][0];
      let y = mmap.choice_locdyn[mmap.choice_locdyn.length - 1][1];
      //budget follow mouse
      let cx = mouse_x - x;
      let cy = mouse_y - y;
      let radians = Math.atan2(cy,cx);
      this.budget_pos_x = x + mmap.budget_dyn[mmap.budget_dyn.length - 1] * Math.cos(radians);
      this.budget_pos_y = y + mmap.budget_dyn[mmap.budget_dyn.length - 1] * Math.sin(radians);
      //draw budget line
      let line = new Phaser.Geom.Line();
      line.setTo(x,y,this.budget_pos_x,this.budget_pos_y);
      this.budget_line.strokeLineShape(line);
    }

    //-------Score--Bar---------------------------------------------------------
    scorebar(mmap){
      //score bar parameters
      this.width = 100 //1000;
      this.height = 400 //480;
      this.box = 12;
      this.top = 50 //200; //distance to screen top

      this.box_center(); //center for labels
      this.incentive(); //calculate incentive: N^2
      this.indicator(mmap); //incentive score indicator, merged with older arrow function
      this.number();
    }

    box_center(){
      this.box_height = this.height / this.box
      this.center_list = []
      this.uni_height = this.box_height / 2
      this.x = this.width / 2 + 600  //larger the number, further to right, 1300

      for (var i=0; i<this.box; i++){
        let loc = [this.x, i * this.box_height + this.uni_height];
        this.center_list.push(loc);
      };
    }

    incentive(){
      this.score = Array.from(Array(this.box).keys());
      this.incentive_score = [];
      for (let i of this.score){
        i = (i**2) * 0.01;
        this.incentive_score.push(i);
      };
    };

    indicator(mmap){
      this.indicator_loc = this.center_list[mmap.n_city[mmap.n_city.length-1]];
      this.indicator_loc_best = this.center_list[Math.max(mmap.n_city)];

      //create triangle arrow and define style
      this.triangle = this.add.graphics();
      this.triangle.fillStyle(grey);

      //arrow parameter
      let point = [this.indicator_loc[0] - 30, this.indicator_loc[1]+this.top+10];
      let v2 = [point[0] - 10, point[1] + 10];
      let v3 = [point[0] - 10, point[1] - 10];
      this.triangle.fillTriangle(point[0], point[1], v2[0],v2[1],v3[0],v3[1]);
    }

    //rendering score Bar
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
    }

//--------Single---Trial--------------------------------------------------------
    // trial(all_done, trl_done, map_content, trl_id, blk, map_id){
    //   trial =
    // };


//--------GAME--LOOP------------------------------------------------------------
//important note for update:
//create general, clear first in update & then redraw again
//clear as a function to update per frame without previous record
    update(time){
      this.add.text(20,20,"Road Construction");
      this.budget_line.clear();
      this.triangle.clear();
      this.line.clear();
      this.draw_map(this.trial,this.input.mousePointer.x,this.input.mousePointer.y);

//the single trial loop need to double check condition, right now only depends on key press
//need to move the following part to single trial function
//need to add static data part for mouse movement
      this.input.on('pointerdown', function (pointer){
        //double check pointer function
        if (pointer.leftButtonDown()){
            if (this.trial.check_end()){
              this.trial.make_choice(this.input.mousePointer.x,this.input.mousePointer.y);
              if (this.trial.check == 1){
                  this.trial.budget_update();
                 this.trial.exp_data([this.input.mousePointer.x,this.input.mousePointer.y],this.input.mousePointer.downTime,1,1,1);
              }else{
                //double check why static here
                  this.trial.static_data([this.input.mousePointer.x,this.input.mousePointer.y],this.input.mousePointer.downTime,1,1,1);
                // console.log('else');
              };
            }else{
              this.add.text(20,50,"Press RETURN to submit");
              //based on key press to change scenes
              this.input.keyboard.on('keydown_ENTER', ()=>this.scene.start('Instruction'));
            };
        };
      }, this);

//if hold the key, it will continue undoing
      if (this.keyZ.isDown && this.trial.choice_dyn[this.trial.choice_dyn.length-1]!=0){
        console.log("Z pressed");
          //haven't code this part of data saving yet and undo draw back
        //this.trial.undo_data([this.input.mousePointer.x,this.input.mousePointer.y],this.keyZ.timeDown,1,1,1);
        // console.log(trial.pos);
        // console.log(trial.time);
      };
    }
}

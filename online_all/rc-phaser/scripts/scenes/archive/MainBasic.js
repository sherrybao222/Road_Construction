/*
This is the first working draft of the RC basic task. This file is later used
to incooperate with undo function. This contains draft for the load function
(need a server to debug this function), for local debug purpose, the current
file use random generated constants value.

Unfinished work include: load function, data saving on mouse and time, trial
and block loop
*/

//import the Map class module (mostly copy pasted, but made some modification 
// for phaser features); need to double check scene parameters in the Map class 
import Map from "../elements/Map.js";

//import some js from Pavlovia lib to enable data saving
import * as data from "../../lib/data-2020.2.js";
import { saveTrialData } from '../util.js';

var trial = 0; //debug purpose

//for current debug purposes 
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

export default class MainTask extends Phaser.Scene {
    constructor() {
        super('MainTask');
    }
    preload()
    {

    }
    create(){
        console.log("Road Basic Ready!");
        //'this' parameter refers to the Phaser Scene
        this.trial = new Map(this,locations,1,1,1); //new trial object from Map Class
        this.draw_map(this.trial,this.input.mousePointer.x,this.input.mousePointer.y);
        //double check the mouse input
    }

    //////////////////Map Visualization////////////////////
    //this call the basic map setup
    draw_map(mmap,mouse_x,mouse_y){
      this.budget(mmap,mouse_x,mouse_y);
      this.cities(mmap);
      this.scorebar(mmap);

      //add the two if statments from python
      // if (mmap.choice_dyn.length >= 2){
      //   this.road(mmap);
      // };

      //need to code to update this condition
      if (mmap.check_end_ind){
        this.add.text(100,400,'You are out of budget');
      };
    }

    //////////////////Functions of map visualization/////////////////////
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

    //////////////////Score Bar functions////////////////////////////////
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


    ///////////////GAME LOOP update per frame///////////////////////////////
//important note for update:
//create general, clear first in update & then redraw again
//clear as a function to update per frame without previous record
    update(){
      this.add.text(20,20,"Road Construction");
      this.budget_line.clear();
      this.triangle.clear(); //wat is this ?? I am lost...
      this.budget(this.trial,this.input.mousePointer.x,this.input.mousePointer.y);
      this.scorebar(this.trial);

      this.input.on('pointerdown', function (pointer){
        //double check pointer function
        if (pointer.leftButtonDown()){
//          console.log(this.trial.choice_dyn);
            if (this.trial.check_end()){
              this.trial.make_choice(this.input.mousePointer.x,this.input.mousePointer.y);
              if (this.trial.check == 1){
                  this.trial.budget_update();
                  this.trial.exp_data(this.pointer,1,1,1,1);
                  this.road(this.trial);
              }else{
                this.trial.static_data(this.pointer,1,1,1,1);
              };
            }else{
              this.add.text(20,50,"Press RETURN to submit");
            //add the trialEnd saving function here
              this.time.delayedCall(1000, trialEnd, [1,1,1], this);
            //it needs to be a callback within this scene's context (maybe is already in the context)
                
              // trialEnd(this, 1,1,1); //this refer to the current scene that contains all the variables 
              //based on key press to change scenes
              // console.log(this.registry.getAll());
              this.input.keyboard.on('keydown_ENTER', ()=>this.scene.start('Instruction'));
            };
        };
      }, this);
    }

}


///////////////Data saving at end of trial///////////////////////////////
var trialEnd = function(blk, trl_id, map_id){
    //change this for RC trial-block flow
    //    if ((trial+1)%repeat == 0){
    //        this.scene.start('MainTask');
    //    }else{
    //        this.scene.start('RCundo');
    //    }
        
    ///this add data to Phaser global data registry
    ///make sure variables are created, this is a template
    this.registry.set("trial"+trial, {blockNo: blk,
                                    trialNo: trl_id,
                                    mapID: map_id,
                                    condition: this.trial.cond,
                                    time: this.time.now,
                                    position: [this.input.mousePointer.x,this.input.mousePointer.y], //need to double check this saving
                                    click: this.trial.click,
                                    undo: this.trial.undo_press,
                                    choiceDyn: this.trial.choice_dyn,
                                    choiceLocDyn: this.trial.choice_locdyn,
                                    choiceHis: this.trial.choice_his,
                                    choiceLoc: this.trial.choice_loc,
                                    budgetDyn: this.trial.budget_dyn,
                                    budgetHis: this.trial.budget_his,
                                    nCity: this.trial.n_city,
                                        });

    /////////this require Pav intergration and debug
    saveTrialData(this.registry.get(`trial${trial}`));
    // psychoJS.experiment.save(); need to debug this
    console.log(this.registry.getAll()); //it's saving, but can be buggy, somehow it's called multiple times
    // trial++;
}
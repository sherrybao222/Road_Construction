/*
This is the first working draft of the RC basic task. This file is later used
to incooperate with undo function. This contains draft for the load function
(need a server to debug this function), for local debug purpose, the current
file use random generated constants value.

Unfinished work include: load function, data saving on mouse and time, trial
and block loop
*/

//import the Map class module (mostly copy pasted, but made some modification 
//for phaser features); need to double check scene parameters in the Map class 
import Map from "../elements/Map.js";

//import some js from Pavlovia lib to enable data saving
// import * as data from "../../lib/data-2020.2.js";
// import { saveTrialData } from '../util.js';



//for current debug purposes 

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
        super('BasicTrain');
    }

    init() {
      // set colors
      this.grey  = 0xFAF7F6;
      this.black = 0x000000;
      this.green = 0xA2EF4C;
      this.mapID = 0; // for test
      this.mapContent = this.registry.values.basicTrain[mapID];
    }

    preload(){
    }
    
    create(){
        console.log("Road Basic Ready!");

        // create map and data saving structure
        this.map = new Map(this.mapContent, this.cameras.main.width, this.cameras.main.height); //new trial object from Map Class
        
        // draw cities
        drawCity(this, this.map, this.black);
        
        // draw budget and move
        this.input.on('pointermove', function (pointer) {
            // define style
            var graphics = this.add.graphics();
            graphics.lineStyle(4, this.green, 1.0);

            //budget follow mouse
            let x = this.map.choiceLocDyn[this.map.choiceLocDyn.length - 1][0];
            let y = this.map.choiceLocDyn[this.map.choiceLocDyn.length - 1][1];
        
            let radians = Math.atan2(pointer.y - y, pointer.x - x);
        
            var budgetPosX = x + this.map.budgetDyn[this.map.budgetDyn.length - 1] * Math.cos(radians);
            var budgetPosY = y + this.map.budgetDyn[this.map.budgetDyn.length - 1] * Math.sin(radians);
        
            //draw budget line
            let line = new Phaser.Geom.Line();
            line.setTo(x, y, budgetPosX, budgetPosY);
            graphics.strokeLineShape(line);

          },this);   

        // draw scorebar
        this.scorebar = new scorebar(this, this.map, this.grey)

        // add title
        this.add.text(20,20,"Road Construction");

    }

    update(){
      this.triangle.clear(); //wat is this ?? I am lost...
      this.scorebar(this.trial);

      this.input.on('pointerdown', function (pointer){
        //double check pointer function
        if (pointer.leftButtonDown()){

            if (this.map.checkEnd()){
              this.map.makeChoice(pointer.x, pointer.y);

              if (this.map.check == 1){
                  this.map.budgetUpdate();
                  this.map.dataChoice(pointer,time); // time
                  drawRoad(this, this.map, this.black)

              }else{
                this.map.dataStatic(pointer, time); // time
              };

            } else {
              this.add.text(20,50,"Press RETURN to submit");
            //add the trialEnd saving function here
            //example: this.time.delayedCall(30000, trialEnd, [], this);
            //it needs to be a callback within this scene's context (maybe is already in the context)
                
              this.trialEnd(1,1,1);
              //based on key press to change scenes
              this.input.keyboard.on('keydown_ENTER', ()=>this.scene.start('Instruction'));
            };
        };
      }, this);
    }
    
    ///////////////Data saving at end of trial///////////////////////////////
    trialEnd(blk,trl_id,map_id){
        //change this for RC trial-block flow
//    if ((trial+1)%repeat == 0){
//        this.scene.start('MainTask');
//    }else{
//        this.scene.start('RCundo');
//    }
    
    var trial = 0; //debug purpose
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
    // saveTrialData(this.registry.get(`trial${trial}`));
    console.log(this.registry.getAll()); //it's saving, but can be buggy, somehow it's called multiple times
    trial++;
    }
}

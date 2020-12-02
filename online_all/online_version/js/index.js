/*
This is the master file that sets up game configuration, and import
different scenes. Phaser scenes are used to organize content, and it uses
a standard template using constructor(); preload();create();update().
More info see resources in Road Construction_QxDq (Google Drive)
*/

//import task scenes 
import Instruction from "./scenes/Instruction.js";
import MainTask from "./scenes/MainTask.js";
import RCundo from "./scenes/RCundo.js";

//game configuration, can export this later if needed 
var config = {
    type: Phaser.AUTO,
    width: 1000,
    height: 600,
    scene: [Instruction,MainTask,RCundo]
  };

var game = new Phaser.Game(config);

//this is the testing code for loading json file
//YES LOADING WORKED HERE, can see info in console 
fetch("./assets/basic_map_training.json")
.then(function(resp){
return resp.json();
})
.then(function(data){
console.log(data)
});

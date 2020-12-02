/*
This is the master file that sets up game configuration, and incooperate
different scenes. Phaser scenes are used to organize content, and it uses
a standard template using constructor(); preload();create();update().
More info see resources in Road Construction_QxDq (Google Drive)
*/

import Instruction from "./scenes/Instruction.js";
import MainTask from "./scenes/MainTask.js";
import RCundo from "./scenes/RCundo.js";

var config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    scene: [Instruction,MainTask,RCundo]
  };

//this set up the canvas and game framework
var game = new Phaser.Game(config);

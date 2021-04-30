/*
This is the master file that sets up game configuration, and import
different scenes. Phaser scenes are used to organize content, and it uses
a standard template using constructor(); preload(); create(); update().
More info see resources in Road Construction_QxDq (Google Drive)
*/

//import task scenes 
//import Preload from "./scenes/Preload.js";
//import EnterID from "./scenes/EnterID.js"; 

import Instruction from "./scenes/Instruction.js";
import BasicTrain from "./scenes/BasicTrain.js";

//import MainBasic from "./scenes/MainBasic.js";
//import MainUndo from "./scenes/MainUndo.js";


// Load our scenes
//var preload = new Preload();
//var enterID = new EnterID();
var instruction = new Instruction();
var basicTrain = new BasicTrain();
//var mainBasic = new MainBasic();
//var mainUndo = new MainUndo()


//set the game configuration 
var config = {
    parent: 'game-container',    //ID of the DOM element to add the canvas to
    type: Phaser.AUTO,
    width: 1000,
    height: 600,
    backgroundColor: "#C8C6C5",
    scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH
      },

    dom: {
        createContainer: true    //to allow text input DOM element
    },

    plugins: {
        scene: [{
            key: 'rexUI',
            plugin: rexuiplugin, //load the UI plugins here for all scenes
            mapping: 'rexUI'
        }]
    },

    audio: {
        disableWebAudio: true
    }

  };

var game = new Phaser.Game(config);

// load scenes
//game.scene.add('Preload', preload);
//game.scene.add("EnterID", enterID);
game.scene.add("Instruction", instruction);
game.scene.add("BasicTrain", basicTrain);
//game.scene.add("MainBasic", mainBasic);
//game.scene.add("MainUndo", mainUndo);

// start 
game.scene.start("BootScene");


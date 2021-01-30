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

//set the game configuration 
var config = {
    type: Phaser.AUTO,
    width: 1000,
    height: 600,
    scene: [Instruction,
            MainTask,
            RCundo
           ], //need to add all scenes here
    parent: 'game-container',    //ID of the DOM element to add the canvas to
    dom: {
        createContainer: true    //to allow text input DOM element
    },
//    plugins: {
//        scene: [{
//            key: 'rexUI',
//            plugin: rexuiplugin, //load the UI plugins here for all scenes
//            mapping: 'rexUI'
//        }]
//    },
    audio: {
        disableWebAudio: true
    }

  };

//set up the canvas and game framework
var game = new Phaser.Game(config);
//console.log(game);
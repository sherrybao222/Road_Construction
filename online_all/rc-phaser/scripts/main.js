/*
This is the master file that sets up game configuration, and import
different scenes. Phaser scenes are used to organize content, and it uses
a standard template using constructor(); preload(); create(); update().
More info see resources in Road Construction_QxDq (Google Drive)
*/

//import task scenes 
// import Preload from "./scenes/Preload.js";
import EnterID from "/js/scenes/EnterID.js"; //this is old foler directory, with new edits on data saving
import MainTask from "/js/scenes/MainTask.js";
import Instruction from "/scripts/scenes/Instruction.js";

// import BasicTrain from "./scenes/BasicTrain.js";
import Undo from "./scenes/Undo.js";

//set the game configuration 
var config = {
    type: Phaser.AUTO,
    width: 1000,
    height: 600,
    scene: [//Preload,
            EnterID,
            Instruction,
            MainTask, //the version that has proper data saving
            // BasicTrain
            //RCundo
           ], //need to add all scenes here
    parent: 'game-container',    //ID of the DOM element to add the canvas to
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

//set up the canvas and game framework
var game = new Phaser.Game(config);

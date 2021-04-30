import BootScene from './scenes/BootScene.js';
import TitleScene from './scenes/TitleScene.js';
import PreloadScene from './scenes/PreloadScene.js';

import GameScene from './scenes/GameScene.js';

// Load our scenes
var bootScene = new BootScene();
var titleScene = new TitleScene();
var preloadScene = new PreloadScene();
var gameScene = new GameScene();


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

    //dom: {
      //  createContainer: true    //to allow text input DOM element
    //}
};


var game = new Phaser.Game(config);

// load scenes
game.scene.add('BootScene', bootScene);
game.scene.add('TitleScene', titleScene);
game.scene.add('PreloadScene', preloadScene);
game.scene.add("GameScene", gameScene);

  // start 
game.scene.start("BootScene");


//scene to initialize experiment and record participant ID. routes to instructions scene.

//import some js from Pavlovia lib to enable data saving
import * as data from "../../lib/data-2020.2.js";
import { PsychoJS } from '../../lib/core-2020.2.js';

//skip built-in error intercept
PsychoJS.prototype._captureErrors = () => {};


//initialise PsychoJS object for saving task data
window.psychoJS = new PsychoJS({ debug: true });   //attached to window object so as to be globally available (across scenes)

//initialize vars
var subID;

//this function extends Phaser.Scene and includes the core logic for the scene
export default class EnterID extends Phaser.Scene {
    constructor() {
        super({
            key: 'EnterID',
            autoStart: true
        });
        
       (async function startPsychoJS() {
       // The experiment handler needed to save our data would
       // be inaccessible before this call resolves. Because of
       // a bug in PsychoJS, please make `expInfo` an empty object
       // instead of skipping if not required
       await psychoJS.start({ expName: 'road construction', expInfo: {} })
       })();
    }

    preload() {
    }
    
    create() {
        //maybe need to change this later for prolific intergration (24 digits ID)
        
        //add popup dialogue box with instructions text
        var instr = this.rexUI.add.dialog({
            background: this.rexUI.add.roundRectangle(0, 0, 400, 400, 20, 0x2F4F4F),
            title: this.rexUI.add.label({
                background: this.rexUI.add.roundRectangle(0, 0, 100, 40, 20, 0x000000),
                text: this.add.text(0, 0, 'Hi!', {
                    fontSize: '24px'
                    }),
                align: 'center',
                space: {
                    left: 15,
                    right: 15,
                    top: 10,
                    bottom: 10
                }
            }),
            content: this.add.text(0, 0, 
                      "Before you start, please enter your Prolific ID.\n\n" +
                      "This is a 3 digit code made up of numbers and letters,\n" +
                      "You can copy this code from your Prolific user profile.\n\n\n\n\n" +  
                      "Please make sure there is no space after you paste the code,\n" +
                      "then click the button below to begin!\n",
                                   
                    {fontSize: "18px",
                     align: 'center'}),
            actions: [
                createLabel(this, 'enter ID')
            ],
            space: {
                title: 25,
                content: 10,
                action: 10,
                left: 10,
                right: 10,
                top: 10,
                bottom: 10,
            },
            align: {
                actions: 'center',
            },
            expand: {
                content: false, 
            }
            });
        
        //control panel position and layout
        var gameHeight = this.sys.game.config.height;
        var gameWidth = this.sys.game.config.width;
        instr
        .setPosition(gameWidth/2, gameHeight/2)
        .layout()
        .popUp(500);
        
        //add text input zone:
        var printText = this.add.text(gameWidth/2, gameHeight/2+20, '', {
                                      fixedWidth: 300, 
                                      fixedHeight: 36,
                                      fontSize: '18px',
                                      color: '#000000',
                                      backgroundColor:  '#ffffff',
                                      align: 'center'
                                      })
        .setOrigin(0.5, 0.5)
        .setInteractive().on('pointerdown', () => {
            this.rexUI.edit(printText)
        });
        
        //control action button functionality (click, hover)
        instr
        .on('button.click', function (button) {
            //here is getting the text input
            subID = printText.text;
            if (subID.length == 3) { 
                printText.setColor('#000000');
                //this.registry.set('subID', subID);  //store ID in data registry
                instr.scaleDownDestroy(500);
                this.nextScene();
            }
            else {
                printText.setColor('#FF0000');
            }
        }, this)
        .on('button.over', function (button) {
            button.getElement('background').setStrokeStyle(2, 0xffffff);
        })
        .on('button.out', function (button) {
            button.getElement('background').setStrokeStyle();
        });
    }
    
    update(time, delta) {
    }
    
    nextScene() {
        window.psychoJS.experiment.addData('subID', subID);
        // console.log(window.psychoJS.experiment);
        this.scene.start('Instruction');
    } 
}

//generic function to create button labels
var createLabel = function (scene, text) {
    return scene.rexUI.add.label({
        background: scene.rexUI.add.roundRectangle(0, 0, 0, 40, 20, 0x778899),
        text: scene.add.text(0, 0, text, {
            fontSize: '20px',
            fill: "#000000"
        }),
        align: 'center',
        width: 40,
        space: {
            left: 10,
            right: 10,
            top: 10,
            bottom: 10
        }
    });
};
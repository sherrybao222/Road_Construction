/*
This contains the instructions for the task extended from the Phaser scene
Class. It use setVisible function to display text.
Not completed flow of instructions
*/

export default class InstructionScene extends Phaser.Scene {
    constructor() {
        super('InstructionScene');
    }

    init(data) {
      this.part  = data.part;
      this.textColor = '#1C2833';
    };

    preload(){
    };

    create(){
      // add title
      if (this.part == 1){
        this.basicInstruction();
      }else {
        this.undoInstruction();
      }
      
      //change scenes on key press command
      this.input.keyboard.on('keydown_ENTER', ()=>this.scene.start('GameScene'));
    };

    update(){
    };

    basicInstruction(){
      var title      = 'Instruction Part 1';
      const screenCenterX = this.cameras.main.worldView.x + this.cameras.main.width / 2;
      this.add.text(screenCenterX, 50, title, { fontFamily: 'Comic Sans MS', fontSize: '37px', fontStyle: 'bold', color: this.textColor, aligh: 'center'}).setOrigin(0.5);

      var text = 'Now you will read the instruction for Road Construction.\nIn Road Construction, you will see a map and a green line as your budget.\nYour goal is to connect as many cities as possible with the given budget.\nThe score bar on the right will show cents you have earned in respect to the number of cities connected.\nPress RETURN to try two examples.'
      this.add.text(50, 200, text,{ fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.textColor, aligh: 'center'});
    }

    undoInstruction(){
      var title      = 'Instruction Part 2';
      const screenCenterX = this.cameras.main.worldView.x + this.cameras.main.width / 2;
      this.add.text(screenCenterX, 50, title, { fontFamily: 'Comic Sans MS', fontSize: '37px', fontStyle: 'bold', color: this.textColor, aligh: 'center'}).setOrigin(0.5);

      var text = 'Now you will read the instruction for Road Construction with Undo.\nIn Road Construction with Undo, you will see a map and a green line as your budget.\nYour goal is to connect as many cities as possible with the given budget.\nIn addition, you can press Z to undo your connections.\nThe score bar on the right will show cents you have earned in respect to the number of cities connected.\nand a record of your highest score achieved.\nPress RETURN to see examples.'
      this.add.text(50, 200, text,{ fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.textColor, aligh: 'center'});
    }
  }

/*
This contains the instructions for the task extended from the Phaser scene
Class. It use setVisible function to display text.
Not completed flow of instructions
*/

export default class Instruction extends Phaser.Scene {
    constructor() {
        super('Instruction');
    }
    preload(){
    };

    create(){
      console.log("instruciton ready!");
      this.trial_start(true);
      //change scenes on key press command
      this.input.keyboard.on('keydown_ENTER', ()=>this.scene.start('RCundo'));
    };

    game_start(blk,show){
      this.gs1 = this.add.text(width/5, height/3, 'This is Part '+ blk + ' on Road Construction',{font: '20px Arial'}).setVisible(show);
      this.gs2 = this.add.text(width/5, height/3 +100, 'Your goal is to connect as many cities as possible with the given budget',{font: '20px Arial'}).setVisible(show);
      this.gs3 = this.add.text(width/5, 550, 'Press RETURN to continue',{font: '20px Arial'}).setVisible(show);
    };

    trial_start(show){
      this.ts1 = this.add.text(50, 200, 'Now you will read the instruction for Road Construction.',{font: '20px Arial'}).setVisible(show);
      this.ts2 = this.add.text(50, 300, 'In Road Construction, you will see a map and a green line as your budget.',{font: '20px Arial'}).setVisible(show);
      this.ts3 = this.add.text(50, 400, 'Your goal is to connect as many cities as possible with the given budget.',{font: '20px Arial'}).setVisible(show);
      this.ts4 = this.add.text(50, 500, 'The score bar on the right will show cents you have earned in respect to the number of cities connected.',{font: '20px Arial'}).setVisible(show);
      this.ts5 = this.add.text(50, 550, 'Press RETURN to continue',{font: '20px Arial'}).setVisible(show);
    };

    post_block(blk,show){
      this.pb1 = this.add.text(width/5, height/3, 'Congratulation, you finished Part '+ blk,{font: '20px Arial'}).setVisible(show);
      this.pb2 = this.add.text(width/5, height/3+100, 'You can take a short break now.',{font: '20px Arial'}).setVisible(show);
      this.pb3 = this.add.text(width/5, 550, 'Press RETURN to continue',{font: '20px Arial'}).setVisible(show);
    };

    update(){

    };
}

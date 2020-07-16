class Instruction extends Phaser.Scene {
    constructor() {
        super('Instruction');
    }
    preload(){
    };

    create(){
      console.log("instruciton ready!");
      // this.keys = this.input.keyboard.createCursorKeys();
      this.keyReturn = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.ENTER);
      this.trial_start(true);
      this.input.on('pointerdown', ()=>this.scene.start('MainTask'));
      // this.input.on('pointerdown',()=>this.game_start(1,true));
      // this.input.on('pointerdown',()=>this.trial_start(true));
    };

    game_start(blk,show){
      this.gs1 = this.add.text(width/5, height/3, 'This is Part '+ blk + ' on Road Construction',{font: '20px Arial'}).setVisible(show);
      this.gs2 = this.add.text(width/5, height/3 +100, 'Your goal is to connect as many cities as possible with the given budget',{font: '20px Arial'}).setVisible(show);
      this.gs3 = this.add.text(width/5, 550, 'Press RETURN to continue',{font: '20px Arial'}).setVisible(show);
    };

    // game_start(blk,show){
    //   const gs1 = this.add.text(width/5, height/3, 'This is Part '+ blk + ' on Road Construction',{font: '20px Arial'}).setVisible(show);
    //   const gs2 = this.add.text(width/5, height/3 +100, 'Your goal is to connect as many cities as possible with the given budget',{font: '20px Arial'}).setVisible(show);
    //   const gs3 = this.add.text(width/5, 550, 'Press RETURN to continue',{font: '20px Arial'}).setVisible(show);
    // };

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
      //the logic flow here is wrong, but key press is working
      // let ins_gs = true;
      // if (ins_gs){
      //   this.input.on('pointerdown',()=>this.game_start(1,true));
      //   ins_gs = false;
      //   this.game_start(1,false);
      // };
      //
      // let ins_ts = true;
      // if (ins_ts){
      //   this.input.on('pointerdown',()=>this.trial_start(true));
      //   ins_ts = false;
      // };
      // // while (ins_gs){
      //   if (this.keyReturn.isDown){
      //     // this.game_start(1,true);
      //     // this.trial_start(true);
      //     // this.scene.restart();
      //     // this.trial_start(true);
      //     console.log('return');
      //   }else{this.game_start(1,false);};
      // // };

      // let ins = true;
      // this.trial_start(true);
      //
      // while (ins){
      //   if (this.keyReturn.isDown){
      //     ins = false;
      //     // this.trial_start(true);
      //     // this.scene.restart();
      //     // this.trial_start(true);
      //     console.log('return');
      //   };
      // };

      // if (this.keyReturn.isDown){
      //   this.game_start(1,false);
      //   this.trial_start(true);
      //   // this.scene.restart();
      //   // this.trial_start(true);
      //   console.log('return');
      // };

      // this.input.keyboard.on('keydown-A', function(){
      //   this.game_start(2,true);
      //   console.log("A down")
      // });
      // if (this.keyReturn.isUp){
      //   this.trial_start(true);
      // };
      // this.input.keyboard.on('keydown-A', function (){
      //   this.trial_start();
      // },this);
      // this.input.keyboard.on('keydown-SPACE', function (){
      //   trial_start();
      // }, this);

      // this.input.keyboard.on('keydown-ENTER', function(){
      //   if (this.keyboard.keydown-ENTER)
      //   this.trial_start();
      // },this);
      // {
      //   //maybe change to button up?
      //   //this.click[-1] = 1
      //   if (this.pointer.leftButtonDown()){
      //       if (this.check_end()){
      //         this.make_choice();
      //         if (this.check == 1){
      //             this.budget_update();
      //             this.exp_data(this.pointer,1,1,1,1);
      //             this.road();
      //         }else{
      //           this.static_data(this.pointer,1,1,1,1);
      //         };
      //       }else{
      //         this.add.text(20,50,"Trial End");
      //       };
      //   };
      // }, this);
    };
}

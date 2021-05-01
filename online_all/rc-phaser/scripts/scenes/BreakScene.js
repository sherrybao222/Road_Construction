export default class BreakScene extends Phaser.Scene {

	constructor() {
        super("BreakScene");
       
    }

    init(data) {
        this.code        = this.registry.values.code; // localStorage.getItem('code') || 

        this.textColor   = '#1C2833';
        this.warnColor   = '#943126';
        
		this.nextObj = this.input.keyboard.addKey('enter');  // Get key object
        this.screenCenterX = this.cameras.main.worldView.x + this.cameras.main.width / 2;

        this.nextBlockInd      = (this.registry.values.trialCounter)/10;
		this.nextOneAll        = this.registry.values.cond[this.nextBlockInd];      
    }

    preload () {
    }

    create () {

        if (this.nextOneAll === 2) {
            var condition = 'Basic';

        } else {
            var condition = 'Undo';
        }

        // add next sign
        this.timedEvent = this.time.addEvent({ delay: this.registry.values.shortBreak, callback: this.next, callbackScope: this});
        var nextSign      = `Congratulations! You have finished ${this.nextBlockInd}/2 of your journey!\nThe next bunch of trials are ${condition} trials.\nYou can have a 5 min break now. When you are ready, press SPACE to continue. \nThe task will automatically continue if you don't press SPACE after 5 min.`;
        
        this.add.text(this.screenCenterX, this.sys.game.config.height-500, nextSign, { fontFamily: 'Comic Sans MS', fontSize: '22px', color: this.textColor, aligh: 'center'}).setOrigin(0.5);
    
    }


    update() {		
        if (this.nextObj.isDown) {	
            this.next()
        }
    }

    next() {
        this.scene.start("GameScene");
    }              
}


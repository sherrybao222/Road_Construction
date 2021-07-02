export default class EndScene extends Phaser.Scene {

	constructor() {
		super("EndScene");
	}
	init() {
        this.code        = this.registry.values.code; // localStorage.getItem('code') || 
		
		this.textColor   = '#1C2833';
		this.warnColor   = '#943126';
		
        this.nextObj     = this.input.keyboard.addKey('enter');  // Get key object
		this.screenCenterX = this.cameras.main.worldView.x + this.cameras.main.width / 2;
		
	}

	create() {
		this.text1 = this.add.text(this.screenCenterX, this.sys.game.config.height-140, '', { fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.warnColor, aligh: 'center'}).setOrigin(0.5);
		this.text2 = this.add.text(this.screenCenterX, this.sys.game.config.height-100, '', { fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.warnColor, aligh: 'center'}).setOrigin(0.5);

		var text = 'The task is finished.\nPress SPACE to save all data.'
		this.add.text(this.screenCenterX, 400, text, { fontFamily: 'Comic Sans MS', fontSize: '30px', fontStyle: 'bold', color: this.textColor, aligh: 'center'}).setOrigin(0.5);
		
	}

    update() {		
        if (this.nextObj.isDown) {	
        }
    }
}
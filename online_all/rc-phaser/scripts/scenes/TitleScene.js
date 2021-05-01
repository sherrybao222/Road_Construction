// after boot scene

export default class TitleScene extends Phaser.Scene {

	constructor() {
		super("TitleScene");
	}

	init() {
		this.textColor   = '#1C2833';
		this.warnColor   = '#943126';

		this.screenCenterX = this.cameras.main.worldView.x + this.cameras.main.width / 2;
		this.nextObj     = this.input.keyboard.addKey('enter');  // Get key object

		// add configurations to registry
		this.validCode   = this.registry.values.validCode;
	}

	create() {

        // add next sign
        var nextSign      = 'After entering your code, press ENTER to start training';
        this.add.text(this.screenCenterX, this.sys.game.config.height-50, nextSign, { fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.textColor, aligh: 'center'}).setOrigin(0.5);
        this.add.text(this.screenCenterX, 300, 'Enter your code:', { fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.textColor }).setOrigin(0.5);;

		// add identifier input
		this.textEntry = this.add.text(this.screenCenterX, 400, '', { fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.textColor}).setOrigin(0.5);;

		this.input.keyboard.on('keydown', function (event) {
			if (event.keyCode === 8 && this.textEntry.text.length > 0) // delete
			{
				this.textEntry.text = this.textEntry.text.substr(0, this.textEntry.text.length - 1);
			}
			else if (event.keyCode >= 48 && event.keyCode < 58) // number 
			{
				this.textEntry.text += event.key;
			}	
		},this);

	}

	update() {		
		if (this.nextObj.isDown) {
            var hasMatch =false;
            for (var index = 0; index < this.validCode.length; ++index) {

                var entry = this.validCode[index];

                if(entry.code == this.textEntry.text){
                hasMatch = true;
                break;
                }
            }
			if (hasMatch)	{	
				this.next();
			} else {
				this.add.text(this.screenCenterX, 500, 'Please enter a valid code!', { fontFamily: 'Comic Sans MS', fontSize: '30px', color: this.warnColor }).setOrigin(0.5);
			}
		}
	}

	next() {
		//localStorage.setItem('code', this.textEntry.text);
		this.registry.set('code', this.textEntry.text);
		this.scene.start("PreloadScene");
	}

}
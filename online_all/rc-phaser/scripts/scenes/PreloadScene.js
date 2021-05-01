export default class PreloadScene extends Phaser.Scene {

	constructor() {
		super('PreloadScene');
	}

	preload() {
		// Add loading screen bars
		this.graphics = this.add.graphics();
		this.newGraphics = this.add.graphics();
		var progressBar = new Phaser.Geom.Rectangle(200, 200, 400, 50);
		var progressBarFill = new Phaser.Geom.Rectangle(205, 205, 290, 40);

		this.graphics.fillStyle(0xffffff, 1);
		this.graphics.fillRectShape(progressBar);

		this.newGraphics.fillStyle(0x3587e2, 1);
		this.newGraphics.fillRectShape(progressBarFill);

		var loadingText = this.add.text(250,260,"Loading: ", { fontFamily: 'Comic Sans MS', fontSize: '32px', fill: '#FFF' });

		this.load.on('progress', this.updateBar, {newGraphics:this.newGraphics,loadingText:loadingText});
		this.load.on('complete', this.complete, {scene:this.scene});

		// load images
		this.load.image('paw', 'scripts/scenes/images/game/paw.png'); //just load something
		
	}

	create() {	
	}

	// ----------------------helper functions-----------------------------

	// show progress
	updateBar(percentage) {
		if(this.newGraphics) {
			this.newGraphics.clear();
			this.newGraphics.fillStyle(0x3587e2, 1);
			this.newGraphics.fillRectShape(new Phaser.Geom.Rectangle(205, 205, percentage*390, 40));
		}
		percentage = percentage * 100;
		this.loadingText.setText("Loading: " + percentage.toFixed(2) + "%");
		console.log("P:" + percentage);
	}

	// scene change
	complete() {
		this.scene.start("InstructionScene", {part: 1});
	}

}
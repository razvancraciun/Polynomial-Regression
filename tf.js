let xs = [];
let ys = [];
let coeffs = [];
const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

window.onload = () => {
	init();
};

Number.prototype.map = function(in_min, in_max, out_min, out_max) {
	return (this - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
};

const init = () => {
	const canvas = document.createElement('canvas');
	canvas.width = '600';
	canvas.height = '600';
	document.body.appendChild(canvas);
	const slider = document.getElementById('coeffs');
	const sliderText = document.getElementById('coeffsText');

	for (let i = 0; i < slider.value; i++) {
		coeffs.push(tf.variable(tf.scalar(Math.random() * 2 - 1)));
	}
	sliderText.innerText = 'Degree: ' + (slider.value - 1);

	slider.onchange = () => {
		for (let i = 0; i < coeffs.length; i++) {
			tf.dispose(coeffs[i]);
		}
		coeffs = [];
		for (let i = 0; i < slider.value; i++) {
			coeffs.push(tf.variable(tf.scalar(Math.random() * 2 - 1)));
		}
		sliderText.innerText = 'Degree: ' + (slider.value - 1);
	};

	draw(canvas);
	canvas.onclick = () => {
		let rect = canvas.getBoundingClientRect();
		let x = event.clientX - rect.left;
		let y = event.clientY - rect.top;

		xs.push(x.map(0, canvas.width, -1, 1));
		ys.push(y.map(0, canvas.height, 1, -1));

		//console.log('x: ' + x.map(0, canvas.width, 0, 1) + ' y: ' + y.map(0, canvas.height, 1, 0));
		setInterval(draw, 10, canvas);
	};
};

const predict = (xs) => {
	const xtensor = tf.tensor1d(xs);
	let ys = tf.scalar(0);
	for (let i = 0; i < coeffs.length; i++) {
		ys = ys.add(xtensor.pow(tf.scalar(i)).mul(coeffs[i]));
	}
	return ys;
};

const loss = (pred, label) => {
	return pred.sub(label).square().mean();
};

const execute = () => {
	tf.tidy(() => {
		let ystensor = tf.tensor1d(ys);
		optimizer.minimize(() => loss(predict(xs), ystensor));
	});
};

const draw = (canvas) => {
	if (xs.length > 0) {
		execute();
	}

	const ctx = canvas.getContext('2d');
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.fillStyle = 'rgba(200, 200, 200, 0.6)';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	for (let i = 0; i < xs.length; i++) {
		ctx.fillStyle = 'rgb(0,0,0)';
		ctx.beginPath();
		ctx.arc(xs[i].map(-1, 1, 0, canvas.width), ys[i].map(-1, 1, canvas.height, 0), 5, 0, 2 * Math.PI);
		ctx.fill();
		ctx.stroke();
	}

	const curveX = [];
	for (let x = -1; x <= 1; x += 0.05) {
		curveX.push(x);
	}
	const y = tf.tidy(() => predict(curveX));

	let curveY = y.dataSync();
	tf.dispose(y);

	ctx.lineWidth = 7;
	ctx.beginPath();
	ctx.moveTo(curveX[0].map(-1, 1, 0, canvas.width), curveY[0].map(-1, 1, canvas.height, 0));
	for (let i = 0; i < curveX.length; i++) {
		let x = curveX[i].map(-1, 1, 0, canvas.width);
		let y = curveY[i].map(-1, 1, canvas.height, 0);
		ctx.lineTo(x, y);
		ctx.moveTo(x, y);
	}

	ctx.stroke();
};

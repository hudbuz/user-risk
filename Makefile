build:
	qwak models build . --model-id credit_risk  --main-dir main

deploy:
	qwak models deploy realtime --model-id credit_risk --pods 1

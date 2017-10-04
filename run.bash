#!/bin/bash
#This is a test script
#This script will execute "$python *.py"
#@date 17/10/04 @author tsmc.fly
#python dnnrun.py #default is execute"$python dnnrun.py -s fcn_dropout_CE_GDO -d MNIST_data -b 200 -l 0.01 -e 50 -D 0.7

fcn_array=(
fcn_dropout_CE_GDO
fcn_dropout_CE_AO
)

batch_array=(
100
200
300
)

epoch_array=(
50
100
)

lr_array=(
0.01
0.05
)


for fcn in ${fcn_array[@]}
do
	for batch in ${batch_array[@]}
	do
		for epoch in ${epoch_array[@]}
		do
			for lr in ${lr_array[@]}
			do
				echo python dnnrun.py -s $fcn -b $batch -e $epoch -l $lr
			done
		done
	done
done 


echo "executing run.bash successful"

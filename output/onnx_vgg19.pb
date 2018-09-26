
A
data_0Placeholder*
dtype0*
shape:àà
B
conv5_1_w_0Const*
valueB€€*
dtype0
5
conv5_3_b_0Const*
value	B€*
dtype0
7
fc6_w_0Const*
valueB€ €Ä*
dtype0
1
fc8_b_0Const*
value	Bè*
dtype0
1
fc7_b_0Const*
value	B€ *
dtype0
B
conv4_3_w_0Const*
valueB€€*
dtype0
5
conv3_2_b_0Const*
value	B€*
dtype0
4
conv1_2_b_0Const*
valueB@*
dtype0
5
conv4_4_b_0Const*
value	B€*
dtype0
@
conv1_1_w_0Const*
valueB@*
dtype0
6
fc7_w_0Const*
valueB
€ € *
dtype0
B
conv3_3_w_0Const*
valueB€€*
dtype0
5
conv4_2_b_0Const*
value	B€*
dtype0
5
conv5_1_b_0Const*
value	B€*
dtype0
5
conv5_4_b_0Const*
value	B€*
dtype0
B
conv5_4_w_0Const*
valueB€€*
dtype0
5
conv4_1_b_0Const*
value	B€*
dtype0
5
conv3_4_b_0Const*
value	B€*
dtype0
B
conv4_4_w_0Const*
valueB€€*
dtype0
B
conv5_3_w_0Const*
valueB€€*
dtype0
B
conv5_2_w_0Const*
valueB€€*
dtype0
6
fc8_w_0Const*
valueB
è€ *
dtype0
1
fc6_b_0Const*
value	B€ *
dtype0
A
conv2_1_w_0Const*
valueB@€*
dtype0
B
conv3_4_w_0Const*
valueB€€*
dtype0
5
conv2_1_b_0Const*
value	B€*
dtype0
B
conv3_2_w_0Const*
valueB€€*
dtype0
B
conv4_2_w_0Const*
valueB€€*
dtype0
@
conv1_2_w_0Const*
valueB@@*
dtype0
B
conv2_2_w_0Const*
valueB€€*
dtype0
5
conv4_3_b_0Const*
value	B€*
dtype0
5
conv2_2_b_0Const*
value	B€*
dtype0
4
conv1_1_b_0Const*
valueB@*
dtype0
B
conv3_1_w_0Const*
valueB€€*
dtype0
5
conv5_2_b_0Const*
value	B€*
dtype0
5
conv3_3_b_0Const*
value	B€*
dtype0
5
conv3_1_b_0Const*
value	B€*
dtype0
B
conv4_1_w_0Const*
valueB€€*
dtype0
E
conv1_1_1/kernelConst*
valueB@*
dtype0
¤
	conv1_1_1Conv2Ddata_0conv1_1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv1_1_2Relu	conv1_1_1*
T0
E
conv1_2_1/kernelConst*
valueB@@*
dtype0
§
	conv1_2_1Conv2D	conv1_1_2conv1_2_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv1_2_2Relu	conv1_2_1*
T0
y
pool1_1MaxPool	conv1_2_2*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

F
conv2_1_1/kernelConst*
valueB@€*
dtype0
¥
	conv2_1_1Conv2Dpool1_1conv2_1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv2_1_2Relu	conv2_1_1*
T0
G
conv2_2_1/kernelConst*
valueB€€*
dtype0
§
	conv2_2_1Conv2D	conv2_1_2conv2_2_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv2_2_2Relu	conv2_2_1*
T0
y
pool2_1MaxPool	conv2_2_2*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

G
conv3_1_1/kernelConst*
valueB€€*
dtype0
¥
	conv3_1_1Conv2Dpool2_1conv3_1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv3_1_2Relu	conv3_1_1*
T0
G
conv3_2_1/kernelConst*
valueB€€*
dtype0
§
	conv3_2_1Conv2D	conv3_1_2conv3_2_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv3_2_2Relu	conv3_2_1*
T0
G
conv3_3_1/kernelConst*
valueB€€*
dtype0
§
	conv3_3_1Conv2D	conv3_2_2conv3_3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv3_3_2Relu	conv3_3_1*
T0
G
conv3_4_1/kernelConst*
valueB€€*
dtype0
§
	conv3_4_1Conv2D	conv3_3_2conv3_4_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv3_4_2Relu	conv3_4_1*
T0
y
pool3_1MaxPool	conv3_4_2*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

G
conv4_1_1/kernelConst*
valueB€€*
dtype0
¥
	conv4_1_1Conv2Dpool3_1conv4_1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv4_1_2Relu	conv4_1_1*
T0
G
conv4_2_1/kernelConst*
valueB€€*
dtype0
§
	conv4_2_1Conv2D	conv4_1_2conv4_2_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv4_2_2Relu	conv4_2_1*
T0
G
conv4_3_1/kernelConst*
valueB€€*
dtype0
§
	conv4_3_1Conv2D	conv4_2_2conv4_3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv4_3_2Relu	conv4_3_1*
T0
G
conv4_4_1/kernelConst*
valueB€€*
dtype0
§
	conv4_4_1Conv2D	conv4_3_2conv4_4_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv4_4_2Relu	conv4_4_1*
T0
y
pool4_1MaxPool	conv4_4_2*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

G
conv5_1_1/kernelConst*
valueB€€*
dtype0
¥
	conv5_1_1Conv2Dpool4_1conv5_1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv5_1_2Relu	conv5_1_1*
T0
G
conv5_2_1/kernelConst*
valueB€€*
dtype0
§
	conv5_2_1Conv2D	conv5_1_2conv5_2_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv5_2_2Relu	conv5_2_1*
T0
G
conv5_3_1/kernelConst*
valueB€€*
dtype0
§
	conv5_3_1Conv2D	conv5_2_2conv5_3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv5_3_2Relu	conv5_3_1*
T0
G
conv5_4_1/kernelConst*
valueB€€*
dtype0
§
	conv5_4_1Conv2D	conv5_3_2conv5_4_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
%
	conv5_4_2Relu	conv5_4_1*
T0
b
pool5_1MaxPool	conv5_4_2*
strides
*
T0*
paddingVALID*
ksize

=
fc6_1MatMulpool5_1fc6_w_0*
transpose_b(*
T0

fc6_2Relufc6_1*
T0
!
fc6_3Identityfc6_2*
T0
;
fc7_1MatMulfc6_3fc7_w_0*
transpose_b(*
T0

fc7_2Relufc7_1*
T0
!
fc7_3Identityfc7_2*
T0
;
fc8_1MatMulfc7_3fc8_w_0*
transpose_b(*
T0
!
prob_1Softmaxfc8_1*
T0
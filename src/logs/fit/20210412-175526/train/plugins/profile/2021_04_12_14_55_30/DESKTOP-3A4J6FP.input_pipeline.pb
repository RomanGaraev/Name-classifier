$	?izF5????;Z?κ???_?Le?!~8gDi??	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails????ò?a??+e??A?E???Ԩ?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?J?4??HP?s?b?A?HP?x?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??~j?t??????Mb??A?~j?t?X?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsa2U0*???????Mbp?AǺ???v?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsa2U0*?????H?}M?A?5?;Nс?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?&S??? ?o_?y?AǺ???f?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsM?O???a??+ei?Ay?&1?|?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???<,Ԋ?"??u??q?A/n????"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???S㥋?/n??b?AM??St$??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	?
F%u??;?O??nr?A	?^)ˀ?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
䃞ͪ?????0?*??A'?W???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailslxz?,C???<,Ԛ?}?A9??v??z?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??6z?{?G?zt?AǺ???V?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?~j?t?x?n??t?A/n??R?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails~8gDi????MbX??A
h"lxz??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???Mb??ݵ?|г??A3ı.n???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???_vOn???_?Le?A/n??R?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???QI???J?4??A??0?*x?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsǺ???v???_?Le?A?~j?t?h?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails{?G?zt?{?G?zd?A{?G?zd?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?]K?=????0?*??A?~j?t?h?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?+e?X????y?):??A{?G?zd?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_?Q?k?/n??b?Aa2U0*?S?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsNbX9????lV}???Avq?-??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?????g?/n??b?AǺ???F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?????̼????V?/??A???B?i??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??6z?/n??r?A????Mb`?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsh??s???/n????A?|гY???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??_?Le??~j?t?X?A/n??R?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?}8gD????ͪ?ն?A??~j?t??*	?????&?@2F
Iterator::ModelB>?٬?'@!w???1?X@)&S??'@1?B\???X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg??j+???!Х㈭D??)n????1???t5*??:Preprocessing2U
Iterator::Model::ParallelMapV22U0*???!???????)2U0*???1???????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<???!??v{????)a??+e??1~?`Z?Ǻ?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??|?5^??!?c_O???)??H?}}?1'?s??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??0?*x?!29?{??)??0?*x?129?{??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!o?!?*P?i??)ŏ1w-!o?1?*P?i??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??A?f??!??:n????)??_?Le?1կ??u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	????gq??NwM;Lƨ???H?}M?!??MbX??	!       "	!       *	!       2$	?v ??|??6??^&??Ǻ???F?!?|гY???:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 
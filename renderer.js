var _debug_pointX = 100.0;
var _debug_pointY = 100.0;
document.addEventListener("keydown", (event) => {
	if (event.key == "l") {_debug_pointX += 1;}
	if (event.key == "k") {_debug_pointY += 1;}
	if (event.key == "j") {_debug_pointX -= 1;}
	if (event.key == "h") {_debug_pointY -= 1;}
})


async function getRenderedImage(device, transform, triangle_data, indicator_transformation) {
  const kTextureWidth = 200;
  const kTextureHeight = 200;
	const cell_count = Math.max(...triangle_data[1].map((x) => Math.max(x[0], x[1]))) + 2;
	const triangle_count = triangle_data[0].length + 2;
	
	function matrix_mult(A, v) {
		let w = A[4][0]*v[0] + A[4][1]*v[1] + A[4][2]*v[2] + A[4][3]*v[3] + A[4][4]

		return [(A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2] + A[0][3]*v[3] + A[0][4])/w,
				(A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2] + A[1][3]*v[3] + A[1][4])/w,
				(A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2] + A[2][3]*v[3] + A[2][4])/w,
				(A[3][0]*v[0] + A[3][1]*v[1] + A[3][2]*v[2] + A[3][3]*v[3] + A[3][4])/w];
	}
	triangles = triangle_data[0].map( (t) => (t.map((v) => matrix_mult(transform, v))))
	triangleInfo = triangle_data[1].map((x) => [x[0], x[1], x[2], x[3]])

	//triangleInfo[2][2] += 16   
	indicator_triangles = [[[0.25, 0.6, 0.25, -1], [0.25, 0.8, 0.3, -1], [0.25, 0.8, 0.2, -1]],
                           [[0.25, 0.6, 0.25, -1], [0.3, 0.8, 0.25, -1], [0.2, 0.8, 0.25, -1]]]
	indicator_triangles = indicator_triangles.map( (t) => t.map((p) => matrix_mult(indicator_transformation, p)))
	triangles = triangles.concat(indicator_triangles)
	triangleInfo.push([cell_count, 0, 7+16, 0])
	triangleInfo.push([cell_count+1, 0, 7+16, 0])
	
	triangleInfo = [].concat(...triangleInfo)
	cellInfo = triangle_data[2];
	//console.log(triangles)
	//triangles.push([[0, 0, 0, 0], [0, 0.5, 0, 1], [0, 0.5, 0, 1]])
	//triangleInfo.push([cell_count-6, 0, 7+16, 0])
	triangles = [].concat(...triangles)
	triangles = [].concat(...triangles)
	const sceneData = [-2.0, 0.25, 0.25, 0.0,
						0.125, 0.125, 0.125, -10.0,
						2.0, 0.0, 0.0, 0.0,
						0.0, 1.0/kTextureWidth, 0.0, 0.0,
						0.0, 0.0, 1.0/kTextureHeight, 0.0,
						Math.round(_debug_pointX), Math.round(_debug_pointY)]
	//console.log(triangle_data)
	const module2 = device.createShaderModule({
	code : `
		struct SceneData {
			origin : vec4<f32>,
			focal_point : vec4<f32>,
			direction : vec4<f32>,
			directionX : vec4<f32>,
			directionY : vec4<f32>,
			debug_pixel : vec2<f32>
		};
		
		struct Triangle {
			A : vec4<f32>,
			B : vec4<f32>,
			C : vec4<f32>,
		};
		
		struct TriangleMetaData {
			left_cell : u32,
			right_cell : u32,
			visibility : u32,
			pad : u32
		}
		
		struct TriangleTemp {
			a_x : vec4<f32>,
			a_y : vec4<f32>,
			a_0 : vec4<f32>,
			b_x : f32,
			b_y : f32,
			b_0 : f32,
			visibility : u32,
			left_cell : u32,
			right_cell : u32,
		};
		
		struct RayPointData {
			x : f32,
			y : f32,
			visibility : f32,
			slope : f32
		};
		
		@group(0) @binding(0) var<storage, read_write> out : array<u32>;
		@group(0) @binding(1) var<storage, read> triangles : array<Triangle>;
		@group(0) @binding(2) var<storage, read> trianglesInfo : array<TriangleMetaData>;
		@group(0) @binding(3) var<uniform> sceneData : SceneData;
		@group(1) @binding(0) var<storage, read_write> out_debug : array<vec2<f32>>;
		@group(1) @binding(1) var<storage, read> cell_color : array<u32>;
		
		var<workgroup> outTriangles : array<TriangleTemp, ${triangle_count}>;
		var<workgroup> intersections : array<RayPointData, ${2*cell_count}>;
		var<workgroup> is_x_above_y : array<array<atomic<u32>, ${Math.ceil(cell_count/32)}>, ${cell_count}>;
		var<workgroup> is_visible : array<atomic<u32>, ${Math.ceil(2*2*cell_count/32)}>;
		var<workgroup> cell_intersections_count : array<atomic<u32>, ${cell_count}>;
		var<workgroup> number_of_barriers : atomic<u32>;
		var<workgroup> seen_thickness : array<atomic<u32>, 3>; //one for each channel
		const width = ${kTextureWidth};
		
				fn cramer_dets(a : mat4x4<f32>, b : vec4<f32>) -> vec4f {
			return vec4f( determinant(mat4x4<f32>(b, a[1], a[2], a[3])),
										determinant(mat4x4<f32>(a[0], b, a[2], a[3])),
										determinant(mat4x4<f32>(a[0], a[1], b, a[3])),
										determinant(mat4x4<f32>(a[0], a[1], a[2], b)));
		} 
		
		fn calculateTriangle(lid : vec3<u32>) {
			var m : mat4x4f = mat4x4f (
													triangles[lid.x].B - triangles[lid.x].A,
													triangles[lid.x].C - triangles[lid.x].A,
													-sceneData.direction,
													sceneData.focal_point - sceneData.origin
													);
			var b : vec4<f32> = sceneData.focal_point - triangles[lid.x].A;
			var a_0 : vec4<f32> = cramer_dets(m, b);
			var b_0 : f32 = determinant(m);
			
			m[2] = -sceneData.directionX;
			var a_x : vec4<f32> = cramer_dets(m, b);
			a_x[2] = 0;
			var b_x : f32 = determinant(m);
			
			m[2] = -sceneData.directionY;
			var a_y : vec4<f32> = cramer_dets(m, b);
			a_y[2] = 0;
		  var b_y : f32 = determinant(m);
			
			outTriangles[lid.x] = TriangleTemp(a_x, a_y, a_0, b_x, b_y, b_0,
															trianglesInfo[lid.x].visibility,
															trianglesInfo[lid.x].left_cell,
															trianglesInfo[lid.x].right_cell);
		}
		fn dump_to(cell : u32, data : RayPointData, wid : vec3<u32>) {
			if (cell > 0){ 
			// cell=0 is placeholder for empty cell, so do not save
			// this means that all other cell are labeled from 1,
			// but we store the from 0, so we save at (cell-1)
			let count = atomicAdd(&cell_intersections_count[cell-1], 1);
			intersections[2*(cell-1) + count] = data;
			//if (count >1){ //TO DEGUB
			//	out[wid.x + width*wid.y] = 123456;
			//}
			}
		}
		
		fn check_is_visible(point_id : u32) -> bool {
			return (atomicLoad(&is_visible[point_id/32]) & (1u << (point_id % 32))) > 0;
		}

		fn check_is_above(x : u32, y : u32) -> bool {
			return (atomicLoad(&is_x_above_y[x][y/32]) & (1u << (y % 32) )) > 0;
		}

		fn darken_by_color(thickness : f32, segment : u32) {
			var factor : f32 = 1000 * thickness /256;
			atomicAdd(&seen_thickness[0], u32(factor * f32(extractBits(cell_color[segment], 0, 8))));
			atomicAdd(&seen_thickness[1], u32(factor * f32(extractBits(cell_color[segment], 8, 8))));
			atomicAdd(&seen_thickness[2], u32(factor * f32(extractBits(cell_color[segment], 16, 8))));
		}
		var<workgroup> is_on_edge : bool;
		var<workgroup> is_overlay : bool;
		@compute @workgroup_size(${triangle_count}) 
		fn computeStuff(@builtin(local_invocation_id) lid : vec3<u32>,
		@builtin(workgroup_id) wid : vec3<u32>,
		@builtin(num_workgroups) wgs : vec3<u32>) {
			//EACH THREAD CORRESPODS TO TRIANGLE
		  calculateTriangle(lid);
			let x = (f32(wid.x) -f32(wgs.x)/2);
			let y = (f32(wid.y) - f32(wgs.y)/2);
			var is_edge : bool = false;
			var is_intersected : bool = false;
			var p_x : f32 = 0.0;
			var p_y : f32 = 0.0;
			let denominator = outTriangles[lid.x].b_x*x + 
			                  outTriangles[lid.x].b_y*y +
												outTriangles[lid.x].b_0;
			let p_u = (outTriangles[lid.x].a_x[0] *x + outTriangles[lid.x].a_y[0]*y +outTriangles[lid.x].a_0[0])/denominator;
			
			if (0 < p_u) {
				let p_v = (outTriangles[lid.x].a_x[1]*x + outTriangles[lid.x].a_y[1]*y +outTriangles[lid.x].a_0[1])/denominator; //maybe it's more efficient to calculate p_u and p_v
				if (0 < p_v && p_u + p_v < 1) {
					//intersection is for sure inside triangle
					is_intersected = true;
					p_x = (outTriangles[lid.x].a_x[2]*x + outTriangles[lid.x].a_y[2]*y +outTriangles[lid.x].a_0[2])/denominator;
					p_y = (outTriangles[lid.x].a_x[3]*x + outTriangles[lid.x].a_y[3]*y +outTriangles[lid.x].a_0[3])/denominator;
					let weight = 0.02;
					is_edge = (((outTriangles[lid.x].visibility & 2)>0) && p_u < weight) ||
									  (((outTriangles[lid.x].visibility & 1)>0) && p_v < weight) ||
										(((outTriangles[lid.x].visibility & 4)>0) && p_u + p_v > 1 -weight);
					
					//if (is_edge) {
					//	is_on_edge = true;
					//out[wid.x + width*wid.y] = 123456;
					//}
					var vis : f32 = 0.0;
					if (is_edge) {
						vis = 1.0;
					}
					//check if the triangle should be flush (i.e. not counted as boundary)
					if ((outTriangles[lid.x].visibility & 16) > 0) {
					   is_overlay = true;
					}
					if ((outTriangles[lid.x].visibility & 8) > 0) { 
						vis = -1.0;
					}
					dump_to(outTriangles[lid.x].left_cell, RayPointData(p_x, p_y, vis, p_x/p_y), wid);
					dump_to(outTriangles[lid.x].right_cell, RayPointData(p_x, p_y, vis, p_x/p_y), wid);
				}
			}
			if (lid.x == 0) {
				atomicStore(&number_of_barriers, 0);
			}
			workgroupBarrier();
			//NOW EACH THREAD CORRESPODS TO A POINT ON THE RAYPLANE
			//if cell_intersections_count is positive that means that cell intersected with RayPlane
			
			var is_blocked : bool = false;
			if (lid.x < ${cell_count} && atomicLoad(&cell_intersections_count[lid.x]) > 1) {
				if (intersections[2*lid.x].slope > intersections[2*lid.x + 1].slope) {
					var _tmp = intersections[2*lid.x + 1];
					intersections[2*lid.x + 1] = intersections[2*lid.x];
					intersections[2*lid.x] = _tmp;
				}
			}
			workgroupBarrier();
			if (lid.x < ${cell_count*2} && atomicLoad(&cell_intersections_count[lid.x / 2]) > (lid.x % 2)) {
				let vis = intersections[lid.x].visibility;
				let p_slope = intersections[lid.x].slope;
				let p_x = intersections[lid.x].x;
				let p_y = intersections[lid.x].y;
				
				for (var cell : u32 = 0; cell < ${cell_count}; cell+=1) { //iterate over segments and check if cur point is blocked
					if (atomicLoad(&cell_intersections_count[cell]) > 1) {
						var k = 2*cell;
						var is_above = intersections[k].slope <= p_slope && p_slope <= intersections[k + 1].slope &&
											(intersections[k + 1].x - intersections[k].x) * (p_y - intersections[k].y) -
											(intersections[k + 1].y - intersections[k].y) * (p_x - intersections[k].x) > 0;
						if (is_above){
							is_blocked = true;
							atomicOr(&is_x_above_y[lid.x/2][cell/32], (1u << (cell%32)));
						}
					}
				}
				
				if(!is_blocked) {atomicOr(&is_visible[lid.x/32], 1u << (lid.x%32));}
				
				if (!is_blocked) {
					if (vis >= 0) {
						atomicAdd(&number_of_barriers, 1);
					}
					if (vis > 0.5) {
						is_on_edge = true;
					}
				}
			}
			workgroupBarrier();
			// calculates visible width of segments
			if ((lid.x < ${2*cell_count}) && (atomicLoad(&cell_intersections_count[lid.x/2]) == 2) &&
			    check_is_visible(lid.x)) {
				var visible_width : f32 = 0;
				var cur_slope = intersections[lid.x].slope;
				if (lid.x % 2 == 0) { //The leftward point of segment
					var best_slope = intersections[lid.x + 1].slope; // if the right end was not visible this would imply existence of better point closer
					for (var i : u32 = 0; i < ${cell_count}; i++) {
						if (check_is_above(lid.x/2, i) && check_is_visible(i*2)) {
							var tmp_slope = intersections[2*i].slope;
							if (cur_slope <= tmp_slope && tmp_slope < best_slope) {best_slope = tmp_slope;}
						} 
					}
					visible_width = best_slope - cur_slope;
					darken_by_color(visible_width, lid.x/2);
				} else { // The rightward edge of segment
				  //find lowest segment
				  var lowest_segment : i32 = -1;
				  for (var seg : u32 = 0; seg < ${cell_count}; seg++) {
				    if ( (atomicLoad(&cell_intersections_count[seg]) > 1) &&
					     (intersections[2*seg].slope < cur_slope) && 
					     (cur_slope < intersections[2*seg + 1].slope)) {
					  if (lowest_segment == -1) {lowest_segment = i32(seg);}
					  if (check_is_above(u32(lowest_segment), seg)) { //TODO: check if this is correct
					    lowest_segment = i32(seg);
	                  }
					} 
				  }
				  if (lowest_segment != -1) {
				  //find closest point
					var best_slope = intersections[lowest_segment*2 + 1].slope; // if the right end was not visible this would imply existence of better point closer
					for (var i : u32 = 0; i < ${cell_count}; i++) {
						if (check_is_visible(i*2)) {
							var tmp_slope = intersections[2*i].slope;
							if (cur_slope <= tmp_slope && tmp_slope < best_slope) {best_slope = tmp_slope;}
						} 
					}
					visible_width = (best_slope - cur_slope)/2; //I assume that each point is attached to 2 segments
					darken_by_color(visible_width, u32(lowest_segment));
	              }
				} 
			}
			workgroupBarrier();
			if (lid.x == 0) {
				//let alpha = 1 - pow(0.9, f32(atomicLoad(&number_of_barriers)));
				//let alpha = 1 - f32(atomicLoad(&number_of_barriers))/20;
				//let brightness = u32(alpha*100);
				let brightness0 = u32(clamp(100 - i32(atomicLoad(&seen_thickness[0])/5), 0, 100));
				let brightness1 = u32(clamp(100 - i32(atomicLoad(&seen_thickness[1])/5), 0, 100));
				let brightness2 = u32(clamp(100 - i32(atomicLoad(&seen_thickness[2])/5), 0, 100));
				let brightness = (brightness0 + brightness1 + brightness2)/3;
				if (is_on_edge){
					out[wid.x + width*wid.y] = brightness;
				} else {
					out[wid.x + width*wid.y] = brightness0 + brightness1*256 + brightness2*256*256;
				}
				if (is_overlay) {
					out[wid.x + width*wid.y] = min(100 + brightness, 255);
				}
				if (u32(sceneData.debug_pixel[0]) == wid.x && u32(sceneData.debug_pixel[1]) == wid.y) {
					out[wid.x + width*wid.y] = 0xFF00FF;
					var position = 0;
					for (var k = 0; k < ${cell_count}; k += 1) {
						if (atomicLoad(&cell_intersections_count[k]) > 0) {
							out_debug[position*2][0] = intersections[k*2].x;
							out_debug[position*2][1] = intersections[k*2].y;
							out_debug[position*2 + 1][0] = intersections[k*2 + 1].x;
							out_debug[position*2 + 1][1] = intersections[k*2 + 1].y;
							position += 1;
						}
					}
				}
			}
		}
		
	`})
	
	const computePipeline = device.createComputePipeline({
    label: 'compute pipeline',
    layout: 'auto',
    compute: {
      module : module2,
			entryPoint: "computeStuff",
    },
  });
	const textureSize = kTextureHeight * kTextureWidth * 4
	
	const triangleBuffer = device.createBuffer({
		label : 'triangles',
		size : 4*(4*3)*triangle_count,
		usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	
	const triangleInfoBuffer = device.createBuffer({
		label : 'triangles',
		size : 4*(4)*triangle_count,
		usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	
	const sceneBuffer = device.createBuffer({
		label : 'scene',
		size : 5*4 + 76,
		usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	const cellInfoBuffer = device.createBuffer({
		label : 'cells',
		size : (3*cell_count)*4,
		usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	const outDebugBuffer = device.createBuffer({
		label : 'outDebug',
		size  : (2*cell_count)*2*4,
		usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const outDebugReadBuffer = device.createBuffer({
		label : 'outDebugRead',
		size  : (2*cell_count)*2*4,
		usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	
	const textureBuffer = device.createBuffer({
			size : textureSize,
			usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});
	
	
	const textureReadBuffer = device.createBuffer({
			size : textureSize,
			usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	
	const bindGroup2 = device.createBindGroup({
		layout : computePipeline.getBindGroupLayout(0),
		entries : [
			{binding : 0 , resource : {buffer : textureBuffer}},
			{binding : 1 , resource : {buffer : triangleBuffer}},
			{binding : 2, resource : {buffer : triangleInfoBuffer}},
			{binding : 3 , resource : {buffer : sceneBuffer}}
		]
	});
	const bindGroup3 = device.createBindGroup({
		layout : computePipeline.getBindGroupLayout(1),
		entries : [
			{binding : 0, resource : {buffer : outDebugBuffer}},
			{binding : 1, resource : {buffer : cellInfoBuffer}}
		]
	})
	

										 
	device.queue.writeBuffer(triangleBuffer, 0, new Float32Array(triangles));
	device.queue.writeBuffer(triangleInfoBuffer, 0, new Int32Array(triangleInfo));
	device.queue.writeBuffer(sceneBuffer, 0, new Float32Array(sceneData));
	device.queue.writeBuffer(cellInfoBuffer, 0, new Uint32Array(cellInfo));

	const encoder = device.createCommandEncoder({label : 'compute builtin encoder'});
	const pass = encoder.beginComputePass({label : 'compute builtin pass'});
	
	pass.setPipeline(computePipeline);
	pass.setBindGroup(0, bindGroup2);
	pass.setBindGroup(1, bindGroup3);
	pass.dispatchWorkgroups(kTextureWidth,kTextureHeight,1)
	pass.end()
	
	encoder.copyBufferToBuffer(textureBuffer, 0, textureReadBuffer, 0, textureSize);
	encoder.copyBufferToBuffer(outDebugBuffer, 0, outDebugReadBuffer, 0, (2*cell_count)*2*4);
	
	device.queue.submit([encoder.finish()])
	await textureReadBuffer.mapAsync(GPUMapMode.READ);
	await outDebugReadBuffer.mapAsync(GPUMapMode.READ);
	return [new Uint8Array(textureReadBuffer.getMappedRange()), 
		    new Float32Array(outDebugReadBuffer.getMappedRange())]
	
}
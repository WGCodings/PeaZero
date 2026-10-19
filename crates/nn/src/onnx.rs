//! Minimal ONNX reader: just enough protobuf to get the nodes and the weight
//! tensors (initializers) out of the graphs `trainer/peazero/model.py`
//! exports, without depending on an ONNX runtime.
//!
//! Protobuf wire format: each field is a varint key `(field << 3) | wire`,
//! wire 0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit.
//! Used fields (onnx.proto3):
//!   ModelProto.graph = 7
//!   GraphProto.node = 1, GraphProto.initializer = 5
//!   NodeProto.input = 1, output = 2, op_type = 4
//!   TensorProto.dims = 1, data_type = 2, float_data = 4, name = 8, raw_data = 9

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, Tensor>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("varint too long".into())
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or("truncated field")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Next field: (field number, wire type, payload for wire 2 / scalar).
    fn field(&mut self) -> Result<(u64, Field<'a>), String> {
        let key = self.varint()?;
        let (num, wire) = (key >> 3, key & 7);
        let f = match wire {
            0 => Field::Varint(self.varint()?),
            1 => Field::Fixed(self.bytes(8)?),
            2 => {
                let n = self.varint()? as usize;
                Field::Bytes(self.bytes(n)?)
            }
            5 => Field::Fixed(self.bytes(4)?),
            w => return Err(format!("unsupported protobuf wire type {w}")),
        };
        Ok((num, f))
    }
}

enum Field<'a> {
    Varint(u64),
    Fixed(&'a [u8]),
    Bytes(&'a [u8]),
}

fn string(b: &[u8]) -> Result<String, String> {
    String::from_utf8(b.to_vec()).map_err(|_| "invalid UTF-8 in ONNX string".to_string())
}

fn parse_node(buf: &[u8]) -> Result<Node, String> {
    let mut r = Reader::new(buf);
    let mut node = Node {
        op_type: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    while !r.done() {
        match r.field()? {
            (1, Field::Bytes(b)) => node.inputs.push(string(b)?),
            (2, Field::Bytes(b)) => node.outputs.push(string(b)?),
            (4, Field::Bytes(b)) => node.op_type = string(b)?,
            _ => {}
        }
    }
    Ok(node)
}

fn parse_tensor(buf: &[u8]) -> Result<(String, Tensor), String> {
    let mut r = Reader::new(buf);
    let (mut name, mut dims, mut data, mut dtype) = (String::new(), Vec::new(), Vec::new(), 1u64);
    while !r.done() {
        match r.field()? {
            (1, Field::Varint(d)) => dims.push(d as usize),
            (1, Field::Bytes(packed)) => {
                let mut p = Reader::new(packed);
                while !p.done() {
                    dims.push(p.varint()? as usize);
                }
            }
            (2, Field::Varint(t)) => dtype = t,
            (4, Field::Bytes(packed)) => data.extend(
                packed
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            (4, Field::Fixed(b)) => data.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            (8, Field::Bytes(b)) => name = string(b)?,
            (9, Field::Bytes(raw)) => {
                data = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            _ => {}
        }
    }
    if dtype != 1 {
        return Err(format!(
            "tensor {name}: only float32 weights are supported (data type {dtype})"
        ));
    }
    let n: usize = dims.iter().product();
    if data.len() != n {
        return Err(format!(
            "tensor {name}: {} values for shape {dims:?}",
            data.len()
        ));
    }
    Ok((name, Tensor { dims, data }))
}

fn parse_graph(buf: &[u8]) -> Result<Graph, String> {
    let mut r = Reader::new(buf);
    let mut g = Graph::default();
    while !r.done() {
        match r.field()? {
            (1, Field::Bytes(b)) => g.nodes.push(parse_node(b)?),
            (5, Field::Bytes(b)) => {
                let (name, t) = parse_tensor(b)?;
                g.initializers.insert(name, t);
            }
            _ => {}
        }
    }
    Ok(g)
}

/// Nodes and initializers of an ONNX model.
pub fn parse_model(bytes: &[u8]) -> Result<Graph, String> {
    let mut r = Reader::new(bytes);
    while !r.done() {
        if let (7, Field::Bytes(b)) = r.field()? {
            return parse_graph(b);
        }
    }
    Err("ONNX model has no graph".into())
}

import React, { useState, useEffect } from 'react';
import { primaryColors, secondaryColors } from "../../utils/colors";
import { InfoCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { 
  Collapse, 
  ConfigProvider, 
  Tooltip, 
  Button, 
  theme, 
  Row, 
  Col, 
  Modal, 
  Select, 
  Input,
  InputNumber, 
  Form, 
  Spin, 
  Table,
  Divider
} from 'antd';

const { Panel } = Collapse;

const formItemLayout = {
  labelCol: {
    xs: { span: 24 },
    sm: { span: 10 },
  },
  wrapperCol: {
    xs: { span: 24 },
    sm: { span: 0 },
  },
};

export default function ESTEFANIA() {
  // Theme
  const {
    token: { colorPrimary },
  } = theme.useToken();

  // Form Hook
  const [form] = Form.useForm();

  // Contact Modal
  const [modal, modalContextHolder] = Modal.useModal();

  // State
  const [ ion, setIon ] = useState([]);
  const [ sources, setSources ] = useState([]);
  const [ template, setTemplate ] = useState([]);
  const [ isLoading, setIsLoading ] = useState(false);
  const [ activeKey, setActiveKey ] = useState(['1', '2']);
  const [ result, setResult ] = useState([]);
  const [ chargeResult, setChargeResult ] = useState('');

  const reset = () => {
    setSources([]);
    setIsLoading(false);
    setActiveKey(['1', '2']);
    setResult([]);
    setChargeResult('');
    form.resetFields();
  };

  useEffect(() => {
    window.api.readTemplate().then((data) => setTemplate(data));
    window.api.readIon().then((data) => setIon(data));
  }, []);

  useEffect(() => {
    if(result.length>1) {
      setChargeResult('test');

      const columns1 = [
        {
          title: 'Fuente',
          dataIndex: 'source',
          key: 'source',
        },
        {
          title: 'ppm',
          dataIndex: 'ppm',
          key: 'ppm',
        }
      ];
      const dataSource1 = result.map((item, index) => {if(item[1]>0) return {key: index, source: item[0], ppm: parseFloat(item[1]*1000).toFixed(2)}}).filter(n => n);
      const dataSource2 = result.map((item) => item[2]).filter(n => n).map((item, index) => <Col key={index} span={6}>{ion[index]}: {item} %</Col>);

      modal.success({
        width: '90%',
        title: 'Solución optimizada con éxito',
        content: <>
                    <Table dataSource={dataSource1} columns={columns1} />
                    <Divider>Error</Divider>
                    <Row>{dataSource2}</Row>
                 </>
      });
    }
  }, [result]);

  // Callbacks
  const showInfo = () => {
    modal.info({
      width: '90%',
      title: 'Bienvenido a ESTEFANIA',
      content: <div style={{textAlign: 'justify'}}>
                  <div>ESTEFANIA (Electronic Salt Target Estimator for Advanced Nutrition in Agriculture) es una aplicación diseñada para ayudarte a calcular y optimizar las fuentes de nutrientes necesarias para tu cultivo. Aquí encontrarás una solución eficiente y precisa para satisfacer las necesidades específicas de tu cultivo en cuanto a nutrientes.</div>
                  <div><br></br>Uso de la aplicación:</div>
                    <ol>
                      <li><b>Ingreso de fuentes de nutrientes:</b> Antes de comenzar, debes ingresar una lista de las fuentes de nutrientes con las que deseas trabajar. De inicio damos soporte con la base de datos que comenzamos a trabajar de manera libre y gratuita.</li>
                      <li><b>Definición de objetivos:</b> Es importante establecer los objetivos deseados para cada elemento o ión de nutriente en tu cultivo. Estos objetivos pueden basarse en las recomendaciones específicas de expertos agrícolas o en tus propias metas. Al definir objetivos claros, ESTEFANIA buscará una solución óptima para satisfacer tus necesidades nutricionales y cumplir con tus metas.</li>
                      <li><b>Cálculo y optimización:</b> Una vez ingresadas las fuentes de nutrientes y los objetivos, ESTEFANIA utilizará un algoritmo avanzado para calcular una solución óptima al problema. El algoritmo tendrá en cuenta todas las restricciones y requerimientos que hayas establecido, y buscará maximizar la eficiencia y minimizar el desperdicio de nutrientes.</li>
                    </ol>
                 <div>Si necesitas más detalles o tienes preguntas específicas, asegúrate de consultar la documentación completa de ESTEFANIA en nuestro<a target="_blank" href="https://github.com/jcrogomez/ESTEFANIA"> repositorio.</a></div> 
                 <div><br></br>En un futuro próximo será posible añadir fuentes de nutrientes diferentes a las que tenemos configuradas, deberás conocer los aportes específicos de cada ion según el manual del proveedor. Esto garantizará resultados más precisos y confiables.</div>
               </div>
    });
  };

  const optimize = () => {
    if(sources.length>=4){
      setIsLoading(true);
      const formValues = {...form.getFieldsValue(), mM: 0, VALOR: 0};
      const mySources = template.map(item => {
        if(sources.includes(item[0])) return {mM: item[0], VALOR: 1};
        else return {mM: item[0], VALOR: 0};
      });
      const columns = [
        "mM",
        "VALOR",
        'HCO3-',
        'NO3-',
        'NH4+',
        'H2PO4-',
        'K+',
        'Ca+2',
        'Mg+2',
        'SO4-2',
        'Fe',
        'Mn',
        'Zn',
        'Cu',
        'B',
        'Mo',
        'Co',
        'Cl-',
        'Na+'
      ];
      window.api.optimize({source: mySources, objective: formValues, columns}).then((data) => {
        console.log('OPTIMIZE RESULTS', data);
        setIsLoading(false);
        setResult(data);
      });
    }else{
      modal.error({
        title: 'Error',
        content: 'Debes seleccionar al menos 4 fuentes para poder utilizar el optimizador'
      });
    }
  };

  const save = () => {
    let filename = '';
    const request = modal.confirm({
      title: 'Selecciona el nombre del archivo',
      content: <Input placeholder='prueba1' onChange={(e) => filename = e.target.value}></Input>,
      footer: <div style={{textAlign:'center', margin:'1rem'}}>
                <Button 
                  type="primary" 
                  onClick={async () => {
                    const status = await window.api.saveFile(filename);
                    request.destroy();
                    if(status===true) reset();
                    else modal.error({title: 'Elige un nombre de archivo válido', content: 'No ingresaste ningún nombre para tu archivo'});
                  }}>
                    Aceptar
                </Button>
              </div>
    });
  };

  const seeResults = () => {
    if(chargeResult==='test'){
      const columns = [
        {
          title: 'Fuente',
          dataIndex: 'source',
          key: 'source',
        },
        {
          title: 'ppm',
          dataIndex: 'ppm',
          key: 'ppm',
        }
      ];
      const dataSource1 = result.map((item, index) => {if(item[1]>0) return {key: index, source: item[0], ppm: parseFloat(item[1]*1000).toFixed(2)}}).filter(n => n);
      const dataSource2 = result.map((item) => item[2]).filter(n => n).map((item, index) => <Col key={index} span={6}>{ion[index]}: {item} %</Col>);

      modal.success({
        width: '90%',
        title: 'Solución optimizada con éxito',
        content: <>
                    <Table dataSource={dataSource1} columns={columns} />
                    <Divider>Error</Divider>
                    <Row>{dataSource2}</Row>
                 </>
      });
    }
    else if(chargeResult === '') {
      const chargeMyResult = (val) => {
        window.api.chargeResult(val).then((data) => {
          const mySources = data.sources;
          const usedSources = data.inputs.slice(1);
          const inputsSources = data.inputs.slice(0,1)[0].slice(2);
          const inputsObjectives = data.inputs.slice(1,2)[0].slice(2);
          
          const columns = [
            {
              title: 'Fuente',
              dataIndex: 'source',
              key: 'source',
            },
            {
              title: 'ppm',
              dataIndex: 'ppm',
              key: 'ppm',
            }
          ];
          const dataSource1 = mySources.map((item, index) => {if(item[1]>0) return {key: index, source: item[0], ppm: parseFloat(item[1]*1000).toFixed(2)}}).filter(n => n);
          const dataSource2 = mySources.map((item) => item[2]).filter(n => n).map((item, index) => <Col key={index} span={6}>{ion[index]}: {item} %</Col>);
  
          const dataSource3 = {};
          inputsSources.forEach((element, index) => {
            dataSource3[element] = inputsObjectives[index];
          });
          form.setFieldsValue(dataSource3);
  
          const dataSource4 = [];
          usedSources.forEach((element, index) => {
            if(element.slice(1, 2)[0]==='1') dataSource4.push(element.slice(0, 1)[0]);
          });
          setSources(dataSource4);
  
          modal.success({
            width: '90%',
            title: 'Solución optimizada con éxito',
            content: <>
                        <Table dataSource={dataSource1} columns={columns} />
                        <Divider>Error</Divider>
                        <Row>{dataSource2}</Row>
                     </>
          });
        });
      }

      window.api.readInput().then((data) => {
        let charge_filename = '';
        const request = modal.success({
          width: '75%',
          title: 'Seleccione un archivo',
          content: <Select
                    style={{ width: '90%' }}
                    placeholder="Selecciona un archivo"
                    onChange={(val) => {setChargeResult(val); charge_filename=val;}}
                    options={data.map((item) => {return {title:item, value:item}})}
                    />,
          footer: <div style={{textAlign:'center', margin:'1rem'}}>
                    <Button type="primary" onClick={() => {request.destroy(); chargeMyResult(charge_filename);}}>
                      Aceptar
                    </Button>
                  </div>
        });
      });
    }
  };

  if(isLoading) return <div style={{textAlign:'center', padding:'2rem'}}><Spin size="large"/></div>;
  return (
      <div style={{ padding: '1rem' }}>
        {modalContextHolder}
        <Row justify={'space-around'} gutter={20} style={{marginBottom:'1rem'}}>
          <Col span={1}>
            <Tooltip title="Ver instrucciones">
              <Button type='ghost' onClick={showInfo}><InfoCircleOutlined style={{color:colorPrimary, fontSize:'1.3rem'}}/></Button>
            </Tooltip>
          </Col>
          <Col span={16} offset={6}>
            <div style={{fontSize:'1.5rem'}}>Optimizador de soluciones nutritivas</div>
          </Col>
          <Col span={1}>
            <Tooltip title="Reset">
              <Button type='ghost' onClick={reset}><DeleteOutlined style={{color:colorPrimary, fontSize:'1.3rem'}}/></Button>
            </Tooltip>
          </Col>
        </Row>
        <ConfigProvider
          theme={{
            token: {
              colorBorder: primaryColors.grey,
              colorFillAlter: secondaryColors.green,
              borderRadiusLG: 20
            },
          }}>
          <Collapse activeKey={activeKey} onChange={(e) => setActiveKey(e)}>
            <Panel header={<b>1. Ingresa tus fuentes de nutrientes</b>} key="1">
              <Select
                  style={{ width: '100%' }}
                  mode="multiple"
                  allowClear
                  placeholder="Selecciona tus fuentes"
                  onChange={(val) => setSources(val)}
                  value={sources}
                  options={template.map((item, index) => {return {title:index, value:item[0]}})}
                  />
            </Panel>
            <Panel header={<b>2. Define tus objetivos</b>} key="2">
              <Form
                  {...formItemLayout}
                  form={form}
                  name="ion"
                  requiredMark={false}
                  layout='horizontal'
              > 
                <Row>
                  {ion.map((i, index) => 
                            <Col key={index} xs={24} s={24} md={12} lg={8} xl={6} xxl={4}>
                              <Form.Item name={i} label={i}>
                                <InputNumber addonAfter='mM/l'/>
                              </Form.Item>
                            </Col>)}
                </Row>
              </Form>
            </Panel>
          </Collapse> 
          <Row justify='space-around' style={{padding: '1rem'}}>
            <Col span={4}>
              <Button type="primary" onClick={optimize}>
                  Optimizar
              </Button>
            </Col>
            <Col span={4}>
              <Button disabled={result.length===0} type="primary" onClick={() => save()}>
                  Guardar
              </Button>
            </Col>
            <Col span={4}>
              <Button type="primary" onClick={seeResults}>
                  {chargeResult==='' ? 'Cargar resultados' : 'Ver resultados'}
              </Button>
            </Col>
          </Row>
        </ConfigProvider>
      </div>
  );
}
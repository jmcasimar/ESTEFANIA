const os = require('os');
const fs = require('fs');
const path = require('path');
const { parse } = require("csv-parse");
const isDev = require('electron-is-dev');
const { stringify } = require("csv-stringify");
const { execFile } = require('child_process');
const { app, BrowserWindow, ipcMain } = require('electron');

// Example of a more typical implementation structure
function createWindow() {
  // Create the browser window.
  const win = new BrowserWindow({
    width: 1200,
    height: 700,
    center: true,
    //minimizable: false,
    //maximizable: false,
    //frame: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: true,
      preload: (isDev
        ? path.join(app.getAppPath(), './public/preload.js') // Loading it from public folder for dev
        : path.join(app.getAppPath(), './build/preload.js')) // Loading it from build folder for production
    },
  });

  // and load the index.html of the app.
  // win.loadFile("index.html");
  win.loadURL(
    isDev
      ? 'http://localhost:3000' // Loading localhost if dev mode
      : `file://${path.join(__dirname, '../build/index.html')}` // Loading build file if in production
  );

  const myIcon = os.platform()==='win32' ? 'favicon.ico' : 'logo512.png';

  win.setIcon(path.join(__dirname, myIcon));
  app.dock.setIcon(path.join(__dirname, myIcon));

  // Open the DevTools.
  if (isDev) {
    win.webContents.openDevTools({ mode: 'detach' });
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
    ipcMain.on('test-send', (event, args) => {
      console.log(fs.readdirSync(__dirname));
    });

    ipcMain.handle('read-inputs', async (event) => {
      try {
        const myPath = isDev ? 'inputs' : path.join(process.resourcesPath, 'inputs');
        const resolvedPath = fs.readdirSync(myPath);
        const fileNames = resolvedPath.map((file) => file.split('.')[0]);
        const filterFiles = fileNames.filter(file => file !== 'test');
        return filterFiles;
      } catch (error) {
        console.error('Error al leer el directorio inputs:', error);
        throw error;
      }
    });

    ipcMain.handle('read-template', async (event) => {
      try {
        const sources = [];
        return new Promise(function (resolve) {
          const myPath = isDev ? path.join('templates', 'iones.csv') : path.join(process.resourcesPath, 'templates', 'iones.csv');
          fs.createReadStream(myPath)
          .pipe(parse({ delimiter: ",", from_line: 2 }))
          .on("data", row => sources.push(row))
          .on("end", () => resolve(sources))
          .on("error", () => console.log(error.message));
        });
      } catch (error) {
        console.error('Error al leer el archivo template/iones.csv', error);
        throw error;
      }
    });

    ipcMain.handle('read-ion', async (event) => {
      try {
        const ion = [];
        return new Promise(function (resolve) {
          const myPath = isDev ? path.join('templates', 'iones.csv') : path.join(process.resourcesPath, 'templates', 'iones.csv');
          fs.createReadStream(myPath)
          .pipe(parse({ delimiter: ",", from_line: 1 }))
          .on("data", row => resolve(row.slice(3)))
          .on("error", () => console.log(error.message));
        });
      } catch (error) {
        console.error('Error al leer el archivo template/iones.csv', error);
        throw error;
      }
    });

    ipcMain.handle('chargeResult', async (event, args) => {
      try {
        const sources = [];
        const inputs = [];
        return new Promise(function (resolve) {
          const myPath1 = isDev ? path.join('outputs', args + '.csv') : path.join(process.resourcesPath, 'outputs', args + '.csv');
          fs.createReadStream(myPath1)
          .pipe(parse({ delimiter: ",", from_line: 2 }))
          .on("data", row => sources.push(row))
          .on("end", () => {
            const myPath2 = isDev ? path.join('inputs', args + '.csv') : path.join(process.resourcesPath, 'inputs', args + '.csv');
            fs.createReadStream(myPath2)
            .pipe(parse({ delimiter: ",", from_line: 1 }))
            .on("data", row => inputs.push(row))
            .on("end", () => resolve({sources, inputs}))
            .on("error", () => console.log(error.message));
          })
          .on("error", () => console.log(error.message));
        });
      } catch (error) {
        console.error('Error al leer el archivo output/' + args + '.csv', error);
        throw error;
      }
    });

    ipcMain.handle('optimize', async (event, args) => {
      const filename = isDev ? path.join('inputs', 'test.csv') : path.join(process.resourcesPath, 'inputs', 'test.csv');
      const writableStream = fs.createWriteStream(filename);
      const stringifier = stringify({ header: true, columns: args.columns });
      stringifier.write(args.objective);
      args.source.forEach(row => stringifier.write(row))
      stringifier.pipe(writableStream);
      if(os.platform()==='win32'){
        try {
          return new Promise(function (resolve) {
            execFile(isDev ? path.join('main.exe') : path.join(process.resourcesPath, 'main.exe'), [isDev ? './' : process.resourcesPath, 'test', 'true'], (error, data) => {
              if(error===null){
                  const result = [];
                    fs.createReadStream(isDev ? path.join('outputs', 'test.csv') : path.join(process.resourcesPath, 'outputs', 'test.csv'))
                    .pipe(parse({ delimiter: ",", from_line: 2 }))
                    .on("data", row => result.push(row))
                    .on("end", () => resolve(result))
                    .on("error", () => console.log(error.message));
              }
              else {
                resolve({error: error, errorMsg: 'Error en la corrida de optimización'});
              }
            });
          });
        } catch(error) {
          console.error('Error al leer el archivo template/iones.csv', error);
          throw error;
        }
      } 
      else if(os.platform()==='darwin') {
        try {
          return new Promise(function (resolve) {
            execFile(isDev ? './main' : path.join(process.resourcesPath, 'main'), [isDev ? './' : process.resourcesPath, 'test', 'true'], (error, data) => {
              if(error===null){
                  const result = [];
                    fs.createReadStream(isDev ? path.join('outputs', 'test.csv') : path.join(process.resourcesPath, 'outputs', 'test.csv'))
                    .pipe(parse({ delimiter: ",", from_line: 2 }))
                    .on("data", row => result.push(row))
                    .on("end", () => resolve(result))
                    .on("error", () => console.log(error.message));
              }
              else {
                resolve({error: error, errorMsg: 'Error en la corrida de optimización'});
              }
            });
          });
        } catch(error) {
          console.error('Error al leer el archivo template/iones.csv', error);
          throw error;
        }
      }
      else {
        console.log('FALTA SOPORTE PARA OTRAS PLATAFORMAS');
        return 'FALTA SOPORTE PARA OTRAS PLATAFORMAS';
      }
    });

    ipcMain.handle('save-file', async (event, args) => {
      if(args==='') {
        return false;
      }
      else {
        fs.rename(isDev ? path.join('inputs', 'test.csv') : path.join(process.resourcesPath, 'inputs', 'test.csv'), isDev ? path.join('inputs', args + '.csv') : path.join(process.resourcesPath, 'inputs', args + '.csv'), (e) => console.log(e));
        fs.rename(isDev ? path.join('outputs', 'test.csv') : path.join(process.resourcesPath, 'outputs', 'test.csv'), isDev ? path.join('outputs', args + '.csv') : path.join(process.resourcesPath, 'outputs', args + '.csv'), (e) => console.log(e));
        return true;
      }
    });

    createWindow();
  
    app.on('activate', function () {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
        console.log('RESET');
      }
    });
  });

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
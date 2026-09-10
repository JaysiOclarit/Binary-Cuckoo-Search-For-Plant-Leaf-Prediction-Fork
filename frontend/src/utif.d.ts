declare module 'utif' {
  export interface TIFFPage {
    width: number;
    height: number;
    [key: string]: unknown;
  }

  export function decode(buffer: ArrayBuffer): TIFFPage[];
  export function decodeImage(buffer: ArrayBuffer, page: TIFFPage): void;
  export function toRGBA8(page: TIFFPage): Uint8Array;
}
